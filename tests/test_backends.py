"""Portability control-flow tests. These do not validate PyTorch or hardware."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Device:
    def __init__(self, name):
        self.name = str(name)
        self.type = self.name.split(":")[0]

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == str(other)


class Tensor:
    def __init__(self, shape, device="cpu", dtype="float16", contiguous=False):
        self.shape = shape
        self.ndim = len(shape)
        self.device = Device(device)
        self.dtype = dtype
        self.is_contiguous = contiguous

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def to(self, device=None, dtype=None):
        return Tensor(self.shape, device or self.device, dtype or self.dtype)

    def contiguous(self):
        self.is_contiguous = True
        return self


@pytest.fixture
def fake_torch(monkeypatch):
    torch = ModuleType("torch")
    torch.nn = ModuleType("torch.nn")
    torch.nn.functional = ModuleType("torch.nn.functional")
    torch.nn.Module = object
    torch.Tensor = Tensor
    torch.device = Device
    torch.float32 = "float32"
    torch.int32 = "int32"
    torch.no_grad = lambda: lambda fn: fn
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch._C = SimpleNamespace(
        _dispatch_has_kernel_for_dispatch_key=Mock(return_value=False)
    )
    torch.ops = SimpleNamespace(
        texture_baker_cpp=SimpleNamespace(
            rasterize=Mock(return_value=Tensor((4, 4, 4))),
            interpolate=Mock(return_value=Tensor((4, 4, 3))),
        )
    )
    for name, value in {
        "torch": torch,
        "torch.nn": torch.nn,
        "torch.nn.functional": torch.nn.functional,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    return torch


@pytest.fixture
def selector(fake_torch, monkeypatch):
    monkeypatch.delenv("SF3D_DEVICE", raising=False)
    monkeypatch.delenv("SF3D_USE_CPU", raising=False)
    module = load_file("device_under_test", "sf3d/device.py")
    module._probe_device = Mock()
    return module


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_device(selector, monkeypatch, device):
    monkeypatch.setenv("SF3D_DEVICE", device)
    selector._is_available = lambda candidate: candidate == device
    assert selector.get_device() == device
    selector._probe_device.assert_called_once_with(device)


def test_auto_tries_accelerators_before_cpu(selector):
    selector._is_available = lambda device: True
    selector._probe_device.side_effect = [RuntimeError("unsupported op"), None]
    assert selector.select_device("auto") == "cpu"
    assert [call.args[0] for call in selector._probe_device.call_args_list] == [
        "cuda",
        "cpu",
    ]


def test_auto_cpu_startup_fallback(selector):
    selector._is_available = lambda device: True
    selector._probe_device.side_effect = [RuntimeError("bad GPU"), None]
    assert selector.select_device("auto") == "cpu"


def test_explicit_device_never_falls_back(selector):
    selector._is_available = lambda device: True
    selector._probe_device.side_effect = RuntimeError("bad GPU")
    with pytest.raises(RuntimeError, match="cuda: bad GPU"):
        selector.select_device("cuda")
    selector._probe_device.assert_called_once_with("cuda")


def test_explicit_unavailable_device(selector):
    with pytest.raises(RuntimeError, match="cuda is not available"):
        selector.select_device("cuda")
    selector._probe_device.assert_not_called()


def test_cpu_probe_failure_is_fatal(selector):
    selector._probe_device.side_effect = ImportError("missing native extension")
    with pytest.raises(RuntimeError, match="missing native extension"):
        selector.select_device("auto")
    selector._probe_device.assert_called_once_with("cpu")


def test_optional_backend_attributes(selector, fake_torch):
    del fake_torch.backends.mps
    assert selector.select_device("auto") == "cpu"


def test_rocm_uses_cuda_name(selector, fake_torch):
    fake_torch.cuda.is_available = lambda: True
    fake_torch.version = SimpleNamespace(hip="6.2")
    assert selector.select_device("auto") == "cuda"


def test_environment_precedence(selector, monkeypatch):
    monkeypatch.setenv("SF3D_USE_CPU", "1")
    assert selector.get_device() == "cpu"
    monkeypatch.setenv("SF3D_DEVICE", "auto")
    selector._is_available = lambda device: True
    assert selector.get_device() == "cpu"
    monkeypatch.setenv("SF3D_DEVICE", " CUDA ")
    selector._is_available = lambda device: True
    assert selector.get_device() == "cuda"


@pytest.mark.parametrize("value", ["rocm", "cuda:0", "mps", "xpu", "", "bogus"])
def test_invalid_device(selector, value):
    with pytest.raises(ValueError, match="SF3D_DEVICE"):
        selector.select_device(value)
    selector._probe_device.assert_not_called()


@pytest.fixture
def baker(fake_torch):
    module = load_file("baker_under_test", "texture_baker/texture_baker/baker.py")
    return module.TextureBaker()


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_baker_cpu_transfers_and_returns_original_device(baker, fake_torch, device):
    result = baker.rasterize(Tensor((3, 2), device), Tensor((1, 3), device), 4)
    assert str(result.device) == device
    uv, indices, resolution = fake_torch.ops.texture_baker_cpp.rasterize.call_args.args
    assert str(uv.device) == str(indices.device) == "cpu"
    assert uv.dtype == "float32" and indices.dtype == "int32"
    assert uv.is_contiguous and indices.is_contiguous and resolution == 4
    result = baker.interpolate(
        Tensor((3, 3), device), Tensor((4, 4, 4), device), Tensor((1, 3), device)
    )
    assert str(result.device) == device
    attr, faces, rast = fake_torch.ops.texture_baker_cpp.interpolate.call_args.args
    assert str(attr.device) == str(faces.device) == str(rast.device) == "cpu"
    assert attr.dtype == rast.dtype == "float32" and faces.dtype == "int32"
    assert attr.is_contiguous and faces.is_contiguous and rast.is_contiguous


def test_baker_native_cuda_keeps_device_index(baker, fake_torch):
    fake_torch._C._dispatch_has_kernel_for_dispatch_key.return_value = True
    baker.rasterize(Tensor((3, 2), "cuda:1"), Tensor((1, 3), "cuda:1"), 4)
    uv, indices, _ = fake_torch.ops.texture_baker_cpp.rasterize.call_args.args
    assert str(uv.device) == str(indices.device) == "cuda:1"


def test_baker_never_retries_runtime_errors(baker, fake_torch):
    fake_torch._C._dispatch_has_kernel_for_dispatch_key.return_value = True
    fake_torch.ops.texture_baker_cpp.rasterize.side_effect = RuntimeError(
        "out of memory"
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        baker.rasterize(Tensor((3, 2), "cuda"), Tensor((1, 3), "cuda"), 4)
    fake_torch.ops.texture_baker_cpp.rasterize.assert_called_once()


def test_baker_forward_passes_only_interpolate_arguments(baker):
    baker.rasterize = Mock(return_value="raster")
    baker.interpolate = Mock(return_value="texture")
    assert baker.forward("attr", "uv", "faces", 4) == "texture"
    baker.interpolate.assert_called_once_with("attr", "raster", "faces")


@pytest.mark.parametrize("path", ["texture_baker/setup.py", "uv_unwrapper/setup.py"])
@pytest.mark.parametrize("hip", [False, True])
def test_extension_build_flags(monkeypatch, fake_torch, path, hip):
    extension = ModuleType("torch.utils.cpp_extension")
    extension.CUDA_HOME = None
    extension.IS_HIP_EXTENSION = hip
    extension.BuildExtension = object
    extension.CppExtension = Mock(side_effect=lambda **kwargs: kwargs)
    extension.CUDAExtension = Mock(side_effect=lambda **kwargs: kwargs)
    setuptools = ModuleType("setuptools")
    setuptools.find_packages = lambda: []
    setuptools.setup = Mock()
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", extension)
    monkeypatch.setitem(sys.modules, "setuptools", setuptools)
    monkeypatch.setattr("platform.system", lambda: "Linux")
    for name in ["USE_CUDA", "USE_METAL", "USE_NATIVE_ARCH", "DEBUG"]:
        monkeypatch.delenv(name, raising=False)
    load_file("setup_under_test", path)
    constructor = extension.CUDAExtension if hip and path.startswith("texture_baker") else extension.CppExtension
    options = constructor.call_args.kwargs
    flags = options["extra_compile_args"]["cxx"]
    assert flags
    assert "-march=native" not in flags
    assert "-fopenmp" in flags
    assert "-fopenmp" in options["extra_link_args"]
    if hip and path.startswith("texture_baker"):
        assert any(source.endswith(".cu") for source in options["sources"])
        assert "libraries" not in options  # CUDAExtension chooses HIP libraries.
    assert all(Path(source).is_file() for source in options["sources"])

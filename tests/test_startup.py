import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from api_backend import Generator


@pytest.fixture
def startup(monkeypatch):
    device = ModuleType("sf3d.device")
    device.get_device = Mock(return_value="cuda")
    system = ModuleType("sf3d.system")
    model = Mock()
    model.eval.return_value = model
    system.SF3D = Mock()
    system.SF3D.from_pretrained.return_value = model
    rembg = ModuleType("rembg")
    rembg.new_session = Mock()
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "rembg", rembg)
    monkeypatch.setitem(sys.modules, "sf3d.device", device)
    monkeypatch.setitem(sys.modules, "sf3d.system", system)
    monkeypatch.delenv("SF3D_DEVICE", raising=False)
    monkeypatch.delenv("SF3D_USE_CPU", raising=False)
    monkeypatch.setattr(Generator, "_probe", Mock())
    return model, rembg


def test_auto_falls_back_only_during_startup(startup):
    model, rembg = startup
    model.to.side_effect = [RuntimeError("GPU allocation failed"), model]
    engine = Generator()
    assert engine.device == "cpu"
    assert [call.args for call in model.to.call_args_list] == [("cuda",), ("cpu",)]
    rembg.new_session.assert_called_once_with(providers=["CPUExecutionProvider"])


def test_explicit_device_does_not_fall_back(startup, monkeypatch):
    model, _ = startup
    monkeypatch.setenv("SF3D_DEVICE", "cuda")
    model.to.side_effect = RuntimeError("GPU allocation failed")
    with pytest.raises(RuntimeError, match="allocation"):
        Generator()
    model.to.assert_called_once_with("cuda")


def test_probe_failure_can_fall_back(startup, monkeypatch):
    monkeypatch.setattr(
        Generator, "_probe", Mock(side_effect=[RuntimeError("kernel failed"), None])
    )
    assert Generator().device == "cpu"


def test_failed_cpu_fallback_aborts_startup(startup):
    model, _ = startup
    model.to.side_effect = RuntimeError("allocation failed")
    with pytest.raises(RuntimeError, match="allocation"):
        Generator()


def test_weight_download_failure_aborts_startup(startup):
    sys.modules["sf3d.system"].SF3D.from_pretrained.side_effect = RuntimeError(
        "access denied"
    )
    with pytest.raises(RuntimeError, match="access denied"):
        Generator()


def test_rembg_failure_aborts_startup(startup):
    _, rembg = startup
    rembg.new_session.side_effect = RuntimeError("cache inaccessible")
    with pytest.raises(RuntimeError, match="cache inaccessible"):
        Generator()

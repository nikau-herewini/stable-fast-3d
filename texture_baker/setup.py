import glob
import os
import platform
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    IS_HIP_EXTENSION,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

library_name = "texture_baker"
root = Path(__file__).resolve().parent


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    # Image builds do not need a GPU. CUDAExtension also handles ROCm/HIP,
    # including its libraries; CUDA_HOME alone incorrectly excludes ROCm.
    has_toolkit = CUDA_HOME is not None or IS_HIP_EXTENSION
    use_cuda = os.getenv("USE_CUDA", "1" if has_toolkit else "0") == "1"
    if use_cuda and not has_toolkit:
        raise RuntimeError("USE_CUDA=1 requires a CUDA or ROCm PyTorch build and toolkit")
    use_metal = os.getenv("USE_METAL", "1" if torch.backends.mps.is_available() else "0") == "1"
    use_native_arch = os.getenv("USE_NATIVE_ARCH", "0") == "1"
    cxx_args = ["-O0" if debug_mode else "-O3", "-fdiagnostics-color=always", "-fopenmp"]
    if use_native_arch:
        cxx_args.append("-march=native")
    compile_args = {"cxx": cxx_args}
    link_args = ["-fopenmp"]
    if debug_mode:
        cxx_args += ["-g", "-UNDEBUG"]
        link_args += ["-O0", "-g"]
        if platform.system() == "Windows":
            cxx_args += ["/Z7", "/Od"]
            link_args.append("/DEBUG")

    sources = glob.glob(str(root / library_name / "csrc" / "*.cpp"))
    macros = []
    if use_cuda:
        sources += glob.glob(str(root / library_name / "csrc" / "*.cu"))
        compile_args["nvcc"] = ["-O0", "-g", "-UNDEBUG"] if debug_mode else ["-O3"]
        macros.append(("THRUST_IGNORE_CUB_VERSION_CHECK", None))
    if use_metal:
        # Keep the upstream Metal build path; the API does not select MPS.
        sources += glob.glob(str(root / library_name / "csrc" / "*.mm"))
        macros.append(("WITH_MPS", None))
        compile_args["cxx"] = ["-O3", "-arch", "arm64", "-mmacosx-version-min=10.15"]
        link_args = ["-arch", "arm64"]

    extension = CUDAExtension if use_cuda else CppExtension
    return [extension(
        name=f"{library_name}._C",
        sources=sources,
        define_macros=macros,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )]


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=[],
    package_data={
        library_name: [os.path.join("csrc", "*.h"), os.path.join("csrc", "*.metal")],
    },
    description="Small texture baker which rasterizes barycentric coordinates to a tensor.",
    long_description=(root / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/Stability-AI/texture_baker",
    cmdclass={"build_ext": BuildExtension},
)

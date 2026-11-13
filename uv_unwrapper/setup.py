import glob
import os
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

library_name = "uv_unwrapper"
root = Path(__file__).resolve().parent


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    is_mac = torch.backends.mps.is_available()
    use_native_arch = not is_mac and os.getenv("USE_NATIVE_ARCH", "0") == "1"
    # Keep the optimization and OpenMP flags even when native tuning is off.
    cxx_args = [
        "-O0" if debug_mode else "-O3",
        "-fdiagnostics-color=always",
        ("-Xclang " if is_mac else "") + "-fopenmp",
    ]
    if use_native_arch:
        cxx_args.append("-march=native")
    if is_mac:
        cxx_args.append("-mmacosx-version-min=10.15")
    link_args = [] if is_mac else ["-fopenmp"]
    if debug_mode:
        cxx_args += ["-g", "-UNDEBUG"]
        link_args += ["-O0", "-g"]

    return [CppExtension(
        name=f"{library_name}._C",
        sources=glob.glob(str(root / library_name / "csrc" / "*.cpp")),
        extra_compile_args={"cxx": cxx_args},
        extra_link_args=link_args,
        libraries=["c10", "torch", "torch_cpu", "torch_python", "omp"] if is_mac else [],
    )]


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=[],
    description="Box projection based UV unwrapper",
    long_description=(root / "README.md").read_text(),
    long_description_content_type="text/markdown",
    cmdclass={"build_ext": BuildExtension},
)

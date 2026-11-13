# syntax=docker/dockerfile:1
# BASE_IMAGE can also be a compatible Debian/Ubuntu PyTorch or toolkit image.
ARG BASE_IMAGE=python:3.11.11-slim-bookworm@sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5
ARG BACKEND=cpu
FROM ${BASE_IMAGE} AS build-base

USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_CUDA=0 \
    USE_METAL=0 \
    USE_NATIVE_ARCH=0 \
    MAX_JOBS=2

# The native extensions need a C++ compiler, OpenMP, Python headers, and torch.
# libGL/glib support OpenCV imported by rembg; libnuma/libdrm support GPU bases.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build pkg-config git ca-certificates \
        python3 python3-dev python3-venv libgomp1 libgl1 libglib2.0-0 \
        libnuma1 libdrm2 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN python -m pip install pip==25.0.1 setuptools==69.5.1 wheel==0.45.1 \
        packaging==24.2 numpy==1.26.4 Pillow==11.3.0

# Only the selected stage is built. GPU wheels supply their user-space runtime;
# the host still needs a supported GPU, driver, and container device access.
FROM build-base AS cpu-torch
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

FROM build-base AS cuda-torch
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN python -m pip install --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

FROM build-base AS rocm-torch
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.2
RUN python -m pip install --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

FROM ${BACKEND}-torch AS api
WORKDIR /opt/sf3d
COPY requirements-inference.txt requirements-api.txt ./
COPY texture_baker/ ./texture_baker/
COPY uv_unwrapper/ ./uv_unwrapper/

# CPU ONNX Runtime handles background removal on every backend. Constrain the
# selected torch ABI and NumPy while resolving the legacy runtime dependencies.
# Install runtime dependencies before compiling either local extension.
RUN python -m pip freeze > /tmp/backend-constraints.txt \
    && python -m pip install -c /tmp/backend-constraints.txt \
        -r requirements-inference.txt -r requirements-api.txt \
    && python -m pip install --no-build-isolation \
        -c /tmp/backend-constraints.txt ./texture_baker/ ./uv_unwrapper/ \
    && python -m pip check \
    && rm /tmp/backend-constraints.txt

COPY sf3d/ ./sf3d/
COPY load/ ./load/
COPY api_server.py api_backend.py LICENSE.md ./
# No gated weights or accelerator is needed for this build-time smoke check.
RUN python - <<'PY'
import torch
from sf3d.device import select_device
from sf3d.system import SF3D
from texture_baker import TextureBaker
assert select_device("cpu") == "cpu"
uv = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
baker = TextureBaker()
rast = baker.rasterize(uv, faces, 8)
mask = baker.get_mask(rast)
texture = baker.interpolate(torch.ones((3, 3)), rast, faces)
assert mask.any().item()
assert texture.shape == (8, 8, 3)
assert torch.allclose(texture[mask], torch.ones_like(texture[mask]))
print("SF3D import, CPU startup, UV unwrap, and texture bake passed")
PY

RUN groupadd --gid 10001 sf3d \
    && useradd --uid 10001 --gid 10001 --create-home sf3d \
    && mkdir -p /cache/huggingface /cache/u2net \
    && chown -R 10001:10001 /cache /home/sf3d
ENV HOME=/home/sf3d \
    HF_HOME=/cache/huggingface \
    U2NET_HOME=/cache/u2net \
    XDG_CACHE_HOME=/cache \
    SF3D_DEVICE=auto
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4).close()"]
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Stable CI/default target. Keep BACKEND=cpu when building this target.
# Accelerator configurations select --target api and set BACKEND explicitly.
FROM api AS cpu

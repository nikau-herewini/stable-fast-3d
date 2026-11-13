"""Startup-only device selection. Inference must use the model's device."""

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _is_available(device: str) -> bool:
    if device == "cpu":
        return True
    return torch.cuda.is_available()


@torch.no_grad()
def _probe_device(device: str) -> None:
    # Import the installed extensions here, not at module import time.
    from texture_baker import TextureBaker
    from uv_unwrapper import Unwrapper

    image = torch.ones((1, 1, 2, 2), dtype=torch.float32, device=device)
    grid = torch.zeros((1, 1, 1, 2), dtype=torch.float32, device=device)
    sampled = F.grid_sample(image, grid, align_corners=True)
    if not torch.allclose(sampled.cpu(), torch.ones((1, 1, 1, 1))):
        raise RuntimeError("grid_sample startup probe failed")

    uv = torch.tensor([[0., 0.], [1., 0.], [0., 1.]], device=device)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    baker = TextureBaker()
    rast = baker.rasterize(uv, faces, 16)
    attr = torch.ones((3, 3), dtype=torch.float32, device=device)
    result = baker.interpolate(attr, rast, faces)
    mask = baker.get_mask(rast)
    if not mask.any().item() or not torch.allclose(result[mask].cpu(), torch.ones_like(result[mask].cpu())):
        raise RuntimeError("texture baker startup probe failed")
    if result.device != uv.device or rast.device != uv.device:
        raise RuntimeError("texture baker returned a different device")

    vertices = torch.tensor(
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        device=device,
    )
    triangles = torch.tensor(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=torch.long, device=device,
    )
    normals = F.normalize(vertices - vertices.mean(0), dim=-1)
    texcoords, indices = Unwrapper()(vertices, normals, triangles, 0.02)
    if not torch.isfinite(texcoords).all().item() or indices.shape != triangles.shape:
        raise RuntimeError("UV unwrapper startup probe failed")


def select_device(requested: Optional[str] = None) -> str:
    """Select and smoke-test a device before model loading.

    ROCm uses the PyTorch ``cuda`` device name. Only ``auto`` can fall back.
    This does not validate model weights, memory capacity, or full inference.
    Do not call this function to retry failed inference requests.
    """
    if requested is None:
        requested = os.environ.get("SF3D_DEVICE")
        if requested is None:
            requested = "cpu" if os.environ.get("SF3D_USE_CPU", "0") == "1" else "auto"
    requested = requested.strip().lower()
    if requested == "auto" and os.environ.get("SF3D_USE_CPU", "0") == "1":
        requested = "cpu"
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("SF3D_DEVICE must be auto, cpu, or cuda (including ROCm)")

    candidates = ("cuda", "cpu") if requested == "auto" else (requested,)
    failures = []
    for device in candidates:
        try:
            if not _is_available(device):
                if requested != "auto":
                    raise RuntimeError(f"{device} is not available in this PyTorch installation")
                continue
            _probe_device(device)
        except Exception as exc:
            failures.append(f"{device}: {exc}")
            if requested != "auto" or device == "cpu":
                raise RuntimeError("SF3D startup device probe failed: " + "; ".join(failures)) from exc
            logger.warning("SF3D startup probe rejected %s: %s", device, exc)
            continue
        logger.info("SF3D selected device: %s", device)
        return device
    raise RuntimeError("No usable SF3D device: " + "; ".join(failures))


def get_device() -> str:
    """Compatibility entry point for startup selection via SF3D_DEVICE."""
    return select_device()

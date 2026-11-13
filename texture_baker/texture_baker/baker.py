import torch
import torch.nn as nn
from torch import Tensor


class TextureBaker(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _operation_device(operation: str, device: torch.device) -> torch.device:
        # CPU-only extension builds can still serve CUDA/ROCm models. Choose
        # the custom-op implementation before execution; never retry failures.
        if device.type in {"cuda", "mps"} and torch._C._dispatch_has_kernel_for_dispatch_key(
            f"texture_baker_cpp::{operation}", device.type.upper()
        ):
            return device
        return torch.device("cpu")

    def rasterize(
        self,
        uv: Tensor,
        face_indices: Tensor,
        bake_resolution: int,
    ) -> Tensor:
        """
        Rasterize the UV coordinates to a barycentric coordinates
        & Triangle idxs texture map

        Args:
            uv (Tensor, num_vertices 2, float): UV coordinates of the mesh
            face_indices (Tensor, num_faces 3, int): Face indices of the mesh
            bake_resolution (int): Resolution of the bake

        Returns:
            Tensor, bake_resolution bake_resolution 4, float: Rasterized map
        """
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError("uv must have shape (num_vertices, 2)")
        if face_indices.ndim != 2 or face_indices.shape[1] != 3:
            raise ValueError("face_indices must have shape (num_faces, 3)")
        if bake_resolution < 1:
            raise ValueError("bake_resolution must be positive")
        if face_indices.numel() == 0:
            result = torch.zeros((bake_resolution, bake_resolution, 4), device=uv.device)
            result[..., -1] = -1
            return result
        operation_device = self._operation_device("rasterize", uv.device)
        return torch.ops.texture_baker_cpp.rasterize(
            uv.to(device=operation_device, dtype=torch.float32).contiguous(),
            face_indices.to(device=operation_device, dtype=torch.int32).contiguous(),
            bake_resolution,
        ).to(uv.device)

    def get_mask(self, rast: Tensor) -> Tensor:
        """
        Get the occupancy mask from the rasterized map

        Args:
            rast (Tensor, bake_resolution bake_resolution 4, float): Rasterized map

        Returns:
            Tensor, bake_resolution bake_resolution, bool: Mask
        """
        return rast[..., -1] >= 0

    def interpolate(
        self,
        attr: Tensor,
        rast: Tensor,
        face_indices: Tensor,
    ) -> Tensor:
        """
        Interpolate the attributes using the rasterized map

        Args:
            attr (Tensor, num_vertices 3, float): Attributes of the mesh
            rast (Tensor, bake_resolution bake_resolution 4, float): Rasterized map
            face_indices (Tensor, num_faces 3, int): Face indices of the mesh
            uv (Tensor, num_vertices 2, float): UV coordinates of the mesh

        Returns:
            Tensor, bake_resolution bake_resolution 3, float: Interpolated attributes
        """
        if attr.ndim != 2 or attr.shape[1] != 3:
            raise ValueError("attr must have shape (num_vertices, 3)")
        if face_indices.ndim != 2 or face_indices.shape[1] != 3:
            raise ValueError("face_indices must have shape (num_faces, 3)")
        if rast.ndim != 3 or rast.shape[2] != 4:
            raise ValueError("rast must have shape (height, width, 4)")
        operation_device = self._operation_device("interpolate", attr.device)
        return torch.ops.texture_baker_cpp.interpolate(
            attr.to(device=operation_device, dtype=torch.float32).contiguous(),
            face_indices.to(device=operation_device, dtype=torch.int32).contiguous(),
            rast.to(device=operation_device, dtype=torch.float32).contiguous(),
        ).to(attr.device)

    def forward(
        self,
        attr: Tensor,
        uv: Tensor,
        face_indices: Tensor,
        bake_resolution: int,
    ) -> Tensor:
        """
        Bake the texture

        Args:
            attr (Tensor, num_vertices 3, float): Attributes of the mesh
            uv (Tensor, num_vertices 2, float): UV coordinates of the mesh
            face_indices (Tensor, num_faces 3, int): Face indices of the mesh
            bake_resolution (int): Resolution of the bake

        Returns:
            Tensor, bake_resolution bake_resolution 3, float: Baked texture
        """
        rast = self.rasterize(uv, face_indices, bake_resolution)
        return self.interpolate(attr, rast, face_indices)

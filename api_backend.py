"""Heavy model dependencies stay out of the HTTP layer and its unit tests."""

import logging
import os

from api_server import InvalidImage

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self):
        import rembg
        import torch

        from sf3d.device import get_device

        self.torch = torch
        self.rembg = rembg
        self.device = get_device()
        # Pin the selection for helpers imported while constructing the model.
        automatic = os.environ.get("SF3D_DEVICE", "auto").strip().lower() == "auto"
        os.environ["SF3D_DEVICE"] = self.device
        from sf3d.system import SF3D

        model = SF3D.from_pretrained(
            os.environ.get("SF3D_MODEL", "stabilityai/stable-fast-3d"),
            config_name="config.yaml",
            weight_name="model.safetensors",
        ).eval()
        try:
            model.to(self.device)
            self._probe(model)
        except Exception:
            if not automatic or self.device == "cpu":
                raise
            logger.warning("GPU startup failed; selecting CPU.", exc_info=True)
            self.device = "cpu"
            os.environ["SF3D_DEVICE"] = "cpu"
            model.to("cpu")
            self._probe(model)
        self.model = model
        # CPU background removal works independently of the inference GPU vendor.
        self.rembg_session = rembg.new_session(providers=["CPUExecutionProvider"])
        logger.info("Model ready on %s", self.device)

    def _probe(self, model):
        """Exercise grid sampling and native texture operators before readiness."""
        torch = self.torch
        with torch.no_grad():
            tensor = torch.ones((1, 1, 2, 2), device=self.device)
            grid = torch.zeros((1, 1, 1, 2), device=self.device)
            torch.nn.functional.grid_sample(tensor, grid, align_corners=True).cpu()
            uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], device=self.device)
            faces = torch.tensor([[0, 1, 2]], device=self.device)
            rast = model.baker.rasterize(uv, faces, 4)
            model.baker.interpolate(
                torch.ones((3, 3), device=self.device), rast, faces
            ).cpu()

    def generate(self, image):
        from sf3d.utils import remove_background, resize_foreground

        foreground = remove_background(image, self.rembg_session)
        try:
            # The crop helper assumes a nonempty, nondegenerate alpha mask.
            with foreground.getchannel("A") as alpha:
                with alpha.point(lambda value: 255 if value > 0 else 0) as mask:
                    bounds = mask.getbbox()
            if bounds is None or bounds[2] - bounds[0] < 2 or bounds[3] - bounds[1] < 2:
                raise InvalidImage("No usable foreground was detected.")
            prepared = resize_foreground(foreground, 0.85)
        finally:
            if foreground is not image:
                foreground.close()
        try:
            with self.torch.inference_mode():
                mesh, _ = self.model.run_image(
                    prepared,
                    bake_resolution=1024,
                    remesh="none",
                    vertex_count=-1,
                )
        finally:
            prepared.close()
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise InvalidImage("No mesh could be reconstructed from the image.")
        return mesh.export(file_type="glb", include_normals=True)

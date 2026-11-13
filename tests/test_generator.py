import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from api_backend import Generator
from api_server import InvalidImage


@pytest.fixture
def generator(monkeypatch):
    utils = ModuleType("sf3d.utils")
    utils.remove_background = Mock(side_effect=lambda image, session: image)
    utils.resize_foreground = Mock(side_effect=lambda image, ratio: image.copy())
    monkeypatch.setitem(sys.modules, "sf3d.utils", utils)
    mesh = SimpleNamespace(
        vertices=[1, 2, 3], faces=[1], export=Mock(return_value=b"glTF")
    )
    engine = Generator.__new__(Generator)
    engine.torch = SimpleNamespace(inference_mode=nullcontext)
    engine.model = SimpleNamespace(run_image=Mock(return_value=(mesh, {})))
    engine.rembg_session = object()
    return engine, utils, mesh


def test_fixed_generation_options(generator):
    engine, utils, mesh = generator
    with Image.new("RGBA", (8, 8), (255, 0, 0, 255)) as image:
        assert engine.generate(image) == b"glTF"
        utils.remove_background.assert_called_once_with(image, engine.rembg_session)
        utils.resize_foreground.assert_called_once_with(image, 0.85)
        assert engine.model.run_image.call_args.kwargs == {
            "bake_resolution": 1024,
            "remesh": "none",
            "vertex_count": -1,
        }
        mesh.export.assert_called_once_with(file_type="glb", include_normals=True)
        assert image.getpixel((0, 0)) == (255, 0, 0, 255)


@pytest.mark.parametrize(
    "size,color", [((8, 8), (0, 0, 0, 0)), ((1, 8), (0, 0, 0, 255))]
)
def test_empty_or_degenerate_foreground(generator, size, color):
    engine, utils, _ = generator
    with Image.new("RGBA", size, color) as image:
        with pytest.raises(InvalidImage, match="foreground"):
            engine.generate(image)
    utils.resize_foreground.assert_not_called()
    engine.model.run_image.assert_not_called()


def test_empty_mesh_is_input_error(generator):
    engine, _, mesh = generator
    mesh.vertices = []
    with Image.new("RGBA", (8, 8), "white") as image:
        with pytest.raises(InvalidImage, match="mesh"):
            engine.generate(image)
    mesh.export.assert_not_called()

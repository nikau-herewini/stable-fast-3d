from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

import api_server


def png(size=(8, 8)):
    data = BytesIO()
    Image.new("RGB", size, "white").save(data, format="PNG")
    return data.getvalue()


def test_file_size_boundary():
    class Engine:
        device = "test"

        def generate(self, image):
            return b"glTF"

    data = png()
    exact = data + b"\x00" * (api_server.MAX_IMAGE_BYTES - len(data))
    with TestClient(api_server.create_app(Engine)) as client:
        assert (
            client.post("/generate", files={"image": ("x", exact)}).status_code == 200
        )
        assert (
            client.post("/generate", files={"image": ("x", exact + b"x")}).status_code
            == 413
        )


def test_pixel_boundary(monkeypatch):
    monkeypatch.setattr(api_server, "MAX_PIXELS", 100)
    with api_server.decode_image(png((10, 10))) as image:
        assert image.size == (10, 10)
    with pytest.raises(api_server.InvalidImage):
        api_server.decode_image(png((11, 10)))


def test_animated_png_rejected():
    data = BytesIO()
    first = Image.new("RGBA", (8, 8), "red")
    second = Image.new("RGBA", (8, 8), "blue")
    first.save(data, format="PNG", save_all=True, append_images=[second])
    with pytest.raises(api_server.InvalidImage, match="Animated"):
        api_server.decode_image(data.getvalue())


def test_compressed_metadata_rejected():
    data = BytesIO()
    info = PngImagePlugin.PngInfo()
    info.add_text("large", "x" * (PngImagePlugin.MAX_TEXT_CHUNK + 1), zip=True)
    Image.new("RGB", (8, 8), "white").save(data, format="PNG", pnginfo=info)
    with pytest.raises(api_server.InvalidImage):
        api_server.decode_image(data.getvalue())


def test_truncated_image_rejected():
    with pytest.raises(api_server.InvalidImage):
        api_server.decode_image(png()[:40])


def test_exif_orientation_applied():
    data = BytesIO()
    image = Image.new("RGB", (8, 4), "white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(data, format="JPEG", exif=exif)
    with api_server.decode_image(data.getvalue()) as result:
        assert result.size == (4, 8)
        assert result.mode == "RGBA"

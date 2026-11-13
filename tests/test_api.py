import asyncio
from io import BytesIO
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api_server
from api_server import create_app

# A valid, empty GLB 2.0 scene: use a fake engine, not gated model weights.
JSON_CHUNK = b'{"asset":{"version":"2.0"},"scene":0,"scenes":[{}]}'
JSON_CHUNK += b" " * (-len(JSON_CHUNK) % 4)
GLB = (
    b"glTF"
    + (2).to_bytes(4, "little")
    + (20 + len(JSON_CHUNK)).to_bytes(4, "little")
    + len(JSON_CHUNK).to_bytes(4, "little")
    + b"JSON"
    + JSON_CHUNK
)


def image_bytes(mode="RGB", color="white", size=(32, 32), fmt="PNG"):
    result = BytesIO()
    Image.new(mode, size, color).save(result, format=fmt)
    return result.getvalue()


class Engine:
    device = "test"

    def generate(self, image):
        assert image.mode == "RGBA"
        return GLB


@pytest.fixture
def client():
    with TestClient(create_app(Engine)) as client:
        yield client


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready", "device": "test"}


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_glb_download(client, fmt):
    response = client.post(
        "/generate", files={"image": ("image", image_bytes(fmt=fmt))}
    )
    assert response.status_code == 200
    assert response.content == GLB
    assert response.headers["content-type"] == "model/gltf-binary"
    assert "model.glb" in response.headers["content-disposition"]
    assert response.headers["cache-control"] == "no-store"


def test_preserve_alpha():
    image = api_server.decode_image(image_bytes("RGBA", (200, 100, 0, 80)))
    assert image.getextrema()[3] == (80, 80)


@pytest.mark.parametrize(
    "files",
    [
        {},
        {"wrong": ("x", b"x")},
        {"image": ("x", b"not an image")},
        [("image", ("a", b"x")), ("image", ("b", b"x"))],
        {"image": ("x", image_bytes(fmt="GIF"))},
        {"image": ("x", image_bytes(size=(1, 32)))},
    ],
)
def test_bad_upload(client, files):
    assert client.post("/generate", files=files).status_code in {400, 415, 422}
    assert (
        client.post("/generate", files={"image": ("x", image_bytes())}).status_code
        == 200
    )


def test_reject_text_fields(client):
    response = client.post(
        "/generate", files={"image": ("x", image_bytes())}, data={"option": "value"}
    )
    assert response.status_code == 400


def test_request_size_limit(client, monkeypatch):
    monkeypatch.setattr(api_server, "MAX_REQUEST_BYTES", 100)
    assert client.post("/generate", content=b"x" * 101).status_code == 413


def test_pixel_limit(client, monkeypatch):
    monkeypatch.setattr(api_server, "MAX_PIXELS", 100)
    assert (
        client.post("/generate", files={"image": ("x", image_bytes())}).status_code
        == 422
    )


def test_error_does_not_expose_details_or_hold_slot():
    class Broken(Engine):
        def generate(self, image):
            raise RuntimeError("secret internal path")

    with TestClient(create_app(Broken)) as client:
        for _ in range(2):
            response = client.post("/generate", files={"image": ("x", image_bytes())})
            assert response.status_code == 500
            assert "secret" not in response.text


def test_startup_failure():
    def fail():
        raise RuntimeError("weights unavailable")

    with pytest.raises(RuntimeError, match="weights unavailable"):
        with TestClient(create_app(fail)):
            pass


def test_busy_rejected_before_reading_body():
    async def check():
        started, release = Event(), Event()

        class Slow(Engine):
            def generate(self, image):
                started.set()
                assert release.wait(10)
                return GLB

        app = create_app(Slow)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(
                    client.post("/generate", files={"image": ("x", image_bytes())})
                )
                try:
                    for _ in range(200):
                        if started.is_set():
                            break
                        await asyncio.sleep(0.01)
                    assert started.is_set()

                    async def unread_body():
                        raise AssertionError("Busy upload body must not be read")
                        yield b""

                    busy = await client.post("/generate", content=unread_body())
                    assert busy.status_code == 503
                    assert busy.headers["retry-after"] == "5"
                    assert (await client.get("/healthz")).status_code == 200
                finally:
                    release.set()
                assert (await first).content == GLB
                assert (
                    await client.post(
                        "/generate", files={"image": ("x", image_bytes())}
                    )
                ).status_code == 200

    asyncio.run(check())


def test_chunked_size_limit(monkeypatch):
    monkeypatch.setattr(api_server, "MAX_REQUEST_BYTES", 100)

    async def check():
        app = create_app(Engine)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:

                async def chunks():
                    yield b"x" * 60
                    yield b"x" * 60

                assert (
                    await client.post("/generate", content=chunks())
                ).status_code == 413

    asyncio.run(check())

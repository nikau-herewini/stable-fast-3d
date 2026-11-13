import asyncio
from io import BytesIO
from threading import Event

import httpx
import pytest
from PIL import Image

import api_server
from api_server import create_app


def png():
    data = BytesIO()
    Image.new("RGB", (8, 8), "white").save(data, format="PNG")
    return data.getvalue()


def test_cancelled_request_holds_slot_until_worker_finishes():
    async def check():
        started, release, finished = Event(), Event(), Event()

        class Slow:
            device = "test"

            def generate(self, image):
                started.set()
                assert release.wait(10)
                finished.set()
                return b"glTF"

        app = create_app(Slow)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                first = asyncio.create_task(
                    client.post("/generate", files={"image": ("x", png())})
                )
                try:
                    for _ in range(200):
                        if started.is_set():
                            break
                        await asyncio.sleep(0.01)
                    assert started.is_set()
                    first.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await first
                    assert not finished.is_set()
                    assert (
                        await client.post("/generate", files={"image": ("x", png())})
                    ).status_code == 503
                finally:
                    release.set()
                for _ in range(200):
                    response = await client.post(
                        "/generate", files={"image": ("x", png())}
                    )
                    if response.status_code != 503:
                        break
                    await asyncio.sleep(0.01)
                assert response.status_code == 200

    asyncio.run(check())


def test_stalled_upload_times_out_and_releases_slot(monkeypatch):
    monkeypatch.setattr(api_server, "UPLOAD_TIMEOUT_SECONDS", 0.02)

    async def check():
        class Engine:
            device = "test"

            def generate(self, image):
                return b"glTF"

        app = create_app(Engine)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:

                async def stalled():
                    yield b"begin"
                    await asyncio.sleep(10)

                assert (
                    await client.post("/generate", content=stalled())
                ).status_code == 408
                assert (
                    await client.post("/generate", files={"image": ("x", png())})
                ).status_code == 200

    asyncio.run(check())


def test_decoder_rejects_metadata_value_errors(monkeypatch):
    monkeypatch.setattr(
        api_server.Image,
        "open",
        lambda _: (_ for _ in ()).throw(ValueError("bad metadata")),
    )
    with pytest.raises(api_server.InvalidImage):
        api_server.decode_image(b"not-empty")

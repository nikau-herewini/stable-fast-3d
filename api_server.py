"""Image-only HTTP interface. Run exactly one worker per model/device."""

import asyncio
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
from threading import Lock
from typing import Callable

import anyio
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 64 * 1024
MAX_PIXELS = 16_000_000
UPLOAD_TIMEOUT_SECONDS = 30


class InvalidImage(ValueError):
    pass


def decode_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise InvalidImage("Image must contain 1 byte to 10 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in {"PNG", "JPEG"}:
                    raise InvalidImage("Only PNG and JPEG images are accepted.")
                if source.width * source.height > MAX_PIXELS:
                    raise InvalidImage("Image exceeds 16 million pixels.")
                if source.width < 2 or source.height < 2:
                    raise InvalidImage("Image dimensions must be at least 2 pixels.")
                if getattr(source, "n_frames", 1) != 1:
                    raise InvalidImage("Animated images are not accepted.")
                source.load()
                return ImageOps.exif_transpose(source).convert("RGBA")
    except InvalidImage:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise InvalidImage("Invalid or oversized image.") from exc


class GenerationSlot:
    """Hold admission until both the HTTP task and its native worker finish."""

    def __init__(self, gate):
        self.gate = gate
        self.mutex = Lock()
        self.users = 1

    def retain(self):
        with self.mutex:
            self.users += 1

    def release(self, _future=None):
        with self.mutex:
            self.users -= 1
            if self.users == 0:
                self.gate.release()


class AdmissionMiddleware:
    """Acquire before reading uploads; bound even chunked HTTP request bodies."""

    def __init__(self, app):
        self.app = app
        self.lock = Lock()

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["path"] != "/generate"
            or scope["method"] != "POST"
        ):
            return await self.app(scope, receive, send)
        if not self.lock.acquire(blocking=False):
            return await JSONResponse(
                {"detail": "Generator is busy."},
                status_code=503,
                headers={"Retry-After": "5"},
            )(scope, receive, send)
        slot = GenerationSlot(self.lock)
        scope["sf3d.slot"] = slot
        try:
            headers = dict(scope["headers"])
            try:
                length = int(headers.get(b"content-length", b"0"))
                if length < 0:
                    raise ValueError
            except ValueError:
                return await JSONResponse({"detail": "Invalid Content-Length."}, 400)(
                    scope, receive, send
                )
            if length > MAX_REQUEST_BYTES:
                return await JSONResponse({"detail": "Upload too large."}, 413)(
                    scope, receive, send
                )
            body = bytearray()
            try:
                with anyio.fail_after(UPLOAD_TIMEOUT_SECONDS):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        chunk = message.get("body", b"")
                        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                            return await JSONResponse(
                                {"detail": "Upload too large."}, 413
                            )(scope, receive, send)
                        body.extend(chunk)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                return await JSONResponse({"detail": "Upload timed out."}, 408)(
                    scope, receive, send
                )
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {
                        "type": "http.request",
                        "body": bytes(body),
                        "more_body": False,
                    }
                return await receive()

            # Keep the slot until response handling and inference both finish.
            await self.app(scope, replay, send)
        finally:
            slot.release()


def create_app(engine_factory: Callable | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        factory = engine_factory
        if factory is None:
            from api_backend import Generator

            factory = Generator
        app.state.engine = await anyio.to_thread.run_sync(factory)
        app.state.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sf3d"
        )
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await anyio.to_thread.run_sync(app.state.executor.shutdown)
            app.state.engine = None

    app = FastAPI(title="Stable Fast 3D API", lifespan=lifespan)
    app.state.ready = False
    app.add_middleware(AdmissionMiddleware)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        if not app.state.ready:
            raise HTTPException(503, "Model is not ready.")
        return {"status": "ready", "device": app.state.engine.device}

    @app.post(
        "/generate",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "model/gltf-binary": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                }
            }
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["image"],
                            "additionalProperties": False,
                            "properties": {
                                "image": {"type": "string", "format": "binary"}
                            },
                        }
                    }
                },
            }
        },
    )
    async def generate(request: Request):
        if not app.state.ready:
            raise HTTPException(503, "Model is not ready.")
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "multipart/form-data"
        ):
            raise HTTPException(415, "Send multipart/form-data with one image field.")
        try:
            async with request.form(max_files=1, max_fields=0) as form:
                items = list(form.multi_items())
                if (
                    len(items) != 1
                    or items[0][0] != "image"
                    or not isinstance(items[0][1], UploadFile)
                ):
                    raise HTTPException(422, "Exactly one image file is required.")
                data = await items[0][1].read(MAX_IMAGE_BYTES + 1)
                if len(data) > MAX_IMAGE_BYTES:
                    raise HTTPException(413, "Image exceeds 10 MiB.")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, "Invalid multipart upload.") from exc

        def run():
            image = decode_image(data)
            try:
                return app.state.engine.generate(image)
            finally:
                image.close()

        try:
            # The concurrent future owns a slot reference. Cancelling the HTTP
            # task cannot release admission while native inference still runs.
            slot = request.scope["sf3d.slot"]
            slot.retain()
            try:
                future = app.state.executor.submit(run)
            except BaseException:
                slot.release()
                raise
            future.add_done_callback(slot.release)
            glb = await asyncio.wrap_future(future)
        except InvalidImage as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            logger.exception("Generation failed")
            raise HTTPException(500, "Generation failed. See server logs.") from exc
        return Response(
            glb,
            media_type="model/gltf-binary",
            headers={
                "Content-Disposition": 'attachment; filename="model.glb"',
                "Cache-Control": "no-store",
            },
        )

    return app


app = create_app()

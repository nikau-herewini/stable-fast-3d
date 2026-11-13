# Image-to-GLB API deployment

This service accepts 1 image and returns 1 GLB. It has **no authentication**.
Keep it on a trusted private network. The examples bind only to `127.0.0.1`.
Do not expose port 8000 directly to the internet. Add an authenticated proxy,
request timeouts, and access controls before any wider deployment.

## CPU quick start

Requirements: Docker with BuildKit, internet access, and
sufficient disk space and RAM for PyTorch, compilation, and model weights.
The intended container platform is Linux x86-64. Builds can use substantial RAM;
`MAX_JOBS=2` limits concurrent native compiler jobs. CPU inference can be slow.

1. Accept the model license and request access at
   <https://huggingface.co/stabilityai/stable-fast-3d>.
2. Create a read token with permission to access that gated repository.
3. Pass the token **at runtime**, not as a build argument:

```bash
# Bash: read without echoing the token or storing it in shell history.
read -rs -p 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
docker build --target cpu -t stable-fast-3d-api:cpu .
docker run -d --name sf3d-api --init --restart unless-stopped \
  --stop-timeout 120 -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN -e SF3D_DEVICE=cpu \
  -v sf3d-model-cache:/cache stable-fast-3d-api:cpu
docker logs -f sf3d-api
```

The server loads its model and background-removal session at startup. Initial
startup downloads weights and can take several minutes. A missing token, denied
model access, incompatible device, or insufficient memory can prevent startup.
No weights or tokens are baked into the image. Container environment variables
are visible to Docker administrators; protect the host and cached model files.

```bash
curl --fail http://127.0.0.1:8000/healthz
curl --fail http://127.0.0.1:8000/readyz
curl --fail-with-body -F 'image=@object.png' \
  http://127.0.0.1:8000/generate --output model.glb
```

With `--fail-with-body`, an error response can be written to `model.glb`.
Check curl's exit status before using the file.

The command is always:

```sh
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

Do not add workers or use reload mode. Each process loads a model and has an
independent admission lock. Scale separate containers only with explicit device
and memory allocation.

## HTTP contract

- `POST /generate`: `multipart/form-data` with exactly 1 file field named `image`.
  No text fields, options, URLs, batch inputs, or additional files are accepted.
- Inputs: single-frame PNG or JPEG; at least 2 pixels per dimension; at most
  **10 MiB** per file and **16,000,000 pixels**. The complete multipart body is
  limited to 10 MiB plus 64 KiB of framing overhead, including chunked uploads.
- Output: `200`, `Content-Type: model/gltf-binary`, attachment `model.glb`, and
  `Cache-Control: no-store`.
- Generation uses a fixed **1024** texture resolution, **0.85** foreground ratio,
  and **no remeshing**. Background removal uses CPU ONNX Runtime even when the
  reconstruction model uses a GPU.
- Only 1 generation request is admitted at a time. Busy requests return **503**
  with `Retry-After: 5`; there is no waiting queue. A disconnected client does not
  free the inference slot while computation is still running.
- Uploads have a total **30-second** deadline; incomplete uploads return `408`.
  Other errors include malformed multipart `400`, upload size `413`, wrong
  content type `415`, invalid image/fields `422`, and inference failure `500`.
  Do not treat an error body as a GLB.
- `GET /healthz`: `{"status":"ok"}` when the HTTP application is serving.
- `GET /readyz`: `{"status":"ready","device":"cpu"}` (or the selected device)
  after model startup; otherwise `503`. Readiness does not mean the generator is
  idle. Uvicorn does not serve HTTP until its startup lifespan completes.

## Cache and process permissions

The image runs as UID/GID **10001:10001**. The named volume
`sf3d-model-cache` persists both `HF_HOME=/cache/huggingface` and
`U2NET_HOME=/cache/u2net`. Recreating the container preserves these downloads.
Removing the container does not remove this volume. Delete `sf3d-model-cache`
only when you intend to discard the cached models.
For bind mounts, create the directory and grant UID 10001 write access first;
an empty root-owned bind mount does not inherit the image's cache ownership.

The image healthcheck calls `/readyz` every 30 seconds, with a 10-minute startup
allowance. Use `docker run --health-start-period=20m` for slower downloads.
Docker marks failed checks unhealthy; it does not automatically restart a
container merely because it is unhealthy. Inspect the logs for startup failures.

## Builds and dependency ordering

```sh
docker build --target cpu -t stable-fast-3d-api:cpu .
# A plain `docker build .` also selects the CPU configuration.
```

The Python base is pinned by version and digest. The backend torch stages pin
PyTorch and torchvision; the dependency files pin the application's direct
requirements. This is **not a complete dependency lock**: transitive Python
packages, apt packages, and downloaded model revisions are not all pinned.

`requirements.txt` remains unchanged for the CLI. Do not install it into the API
image: its non-macOS `rembg[gpu]` extra pulls in NVIDIA ONNX Runtime.
`requirements-inference.txt` preserves the legacy model dependency versions,
including NumPy 1.26.4, but uses plain `rembg==2.0.57` and CPU
`onnxruntime==1.20.1`. `requirements-api.txt` supplies the HTTP dependencies.

The Dockerfile installs torch **first**, then runtime requirements, then
`./texture_baker/` and `./uv_unwrapper/` with **`--no-build-isolation`**. It
constrains the installed torch version so dependency resolution cannot replace
the selected backend. A C++ compiler, Python headers, CMake, Ninja, and OpenMP
runtime are installed. Both extensions are rebuilt for each backend's torch ABI.
`USE_CUDA=0 USE_METAL=0 USE_NATIVE_ARCH=0` selects portable CPU extension builds;
no CUDA or Metal texture-baker kernel is needed. The baker wrapper transfers
its inputs to CPU and returns results to the inference device.

Every image build imports `SF3D` and runs a tiny CPU texture-baker check. This
needs no gated model weights. To repeat a weight-free check in the built image:

```sh
docker run --rm -i --entrypoint python stable-fast-3d-api:cpu - <<'PY'
import torch
from sf3d.system import SF3D
from texture_baker import TextureBaker
uv = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
baker = TextureBaker()
rast = baker.rasterize(uv, faces, 8)
mask = baker.get_mask(rast)
texture = baker.interpolate(torch.ones((3, 3)), rast, faces)
assert mask.any().item()
assert texture.shape == (8, 8, 3)
assert torch.allclose(texture[mask], torch.ones_like(texture[mask]))
print('SF3D import and CPU texture bake passed')
PY
```

## Device selection

`SF3D_DEVICE` accepts `auto` (default), `cpu`, or `cuda`.
ROCm uses the PyTorch name **`cuda`**, not `rocm`. Auto selection probes CUDA/ROCm
at startup and can fall back to CPU. An explicit device fails startup
if unavailable or rejected by its startup probe. The probe is not proof that
full model inference fits in memory or implements every required operation.
Inference errors do not trigger a CPU retry.

The legacy `SF3D_USE_CPU=1` selects CPU when `SF3D_DEVICE` is unset or `auto`.
An explicit `SF3D_DEVICE=cuda` wins over the legacy flag. Prefer
`SF3D_DEVICE=cpu` for a clear CPU configuration. Neither variable can add a
backend absent from the installed PyTorch build.

## GPU image configurations — untested

**Docker builds and GPU/model inference were not executed locally.** Local API
tests passed using a temporary Python environment; Docker and a C++ compiler
were unavailable locally. The CPU image build and weight-free startup, UV
unwrapping, and texture-baker checks passed in CI. GPU configurations remain
untested. These are deployment configurations, not
claims of support for arbitrary NVIDIA or AMD GPUs. Check the selected
PyTorch release's hardware and driver requirements. CPU baking also adds data
transfers and CPU work to accelerator inference.

The final `api` stage selects a named torch base stage using `BACKEND`:

| `BACKEND` | Torch / torchvision | Wheel index |
| --- | --- | --- |
| `cpu` | 2.5.1 / 0.20.1 | `cpu` |
| `cuda` | 2.5.1 / 0.20.1 | `cu124` |
| `rocm` | 2.5.1 / 0.20.1 | `rocm6.2` |

Build GPU configurations with `--target api --build-arg BACKEND=...`.
The default `cpu` target is an alias for this stage with `BACKEND=cpu`.
The wheels use `https://download.pytorch.org/whl/<index>`. Build arguments
`TORCH_VERSION`, `TORCHVISION_VERSION`, and `TORCH_INDEX_URL` can override the
pair/index together; do not change only one without checking compatibility.

`BASE_IMAGE` can replace the default Python image with an apt-based
Debian/Ubuntu PyTorch or toolkit image. It must provide a usable Python
**3.10–3.12** and the backend's runtime libraries. A fresh virtual environment
installs the selected torch wheels; preinstalled vendor torch is not reused.
Other distributions, Python 3.13+, and arbitrary vendor images are not supported
by this recipe. Examples of version-and-digest-pinned starting bases:

- CUDA PyTorch:
  `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014`
- ROCm toolkit:
  `rocm/dev-ubuntu-22.04:6.2.4@sha256:030304ab0a88778b627b2eb2354c419146e105f5ac39ffd9c2438c82ef54b584`

These tags/digests were checked against the registry, not built. A toolkit image
alone does not guarantee the required GPU userspace driver is installed.

### NVIDIA CUDA

Install a host NVIDIA driver compatible with CUDA 12.4 and NVIDIA Container
Toolkit. Select a GPU supported by this PyTorch CUDA build.

```sh
docker build --target api --build-arg BACKEND=cuda \
  -t stable-fast-3d-api:cuda .
docker run --rm --init --gpus all -p 127.0.0.1:8000:8000 \
  -e HF_TOKEN -e SF3D_DEVICE=cuda \
  -v sf3d-model-cache:/cache stable-fast-3d-api:cuda
```

The default Python base plus CUDA wheels does not need nvcc because both native
extensions are CPU builds. To use the pinned CUDA PyTorch base above, additionally
pass `--build-arg BASE_IMAGE='pytorch/pytorch:...@sha256:...'` with its full value.

### AMD ROCm

Use a ROCm-supported Linux host, AMD GPU, and compatible kernel driver. The
PyTorch 2.5.1 wheels target ROCm 6.2; do not infer support for older Radeon cards
or every integrated AMD GPU. `/dev/kfd` and `/dev/dri` must exist on the host.

```sh
docker build --target api --build-arg BACKEND=rocm \
  --build-arg BASE_IMAGE=rocm/dev-ubuntu-22.04:6.2.4@sha256:030304ab0a88778b627b2eb2354c419146e105f5ac39ffd9c2438c82ef54b584 \
  -t stable-fast-3d-api:rocm .
docker run --rm --init -p 127.0.0.1:8000:8000 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  -e HF_TOKEN -e SF3D_DEVICE=cuda \
  -v sf3d-model-cache:/cache stable-fast-3d-api:rocm
```

Adjust the render-node path and supplementary group IDs to the selected host
GPU. Do not solve permissions by running the application privileged or as root.

The API deployment scope is **CPU, NVIDIA CUDA, and AMD ROCm only**. Existing
Mac source is retained, but MPS is outside the API support scope.

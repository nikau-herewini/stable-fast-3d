# Stable Fast 3D API

An API-only fork of [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d)
that reconstructs a textured GLB model from one image.

## API

- `POST /generate`: upload one PNG/JPEG file in the multipart field `image`;
  download the resulting GLB.
- `GET /healthz`: HTTP health.
- `GET /readyz`: model readiness and selected device.

The API has **no authentication**. Keep it on a trusted private network.
Generation settings are fixed; only one request runs at a time. Busy requests
return `503`. Files are limited to 10 MiB and 16 million pixels.

## Docker deployment

Use Linux amd64 images from `ghcr.io/nikau-herewini/stable-fast-3d`:

| Tag | Inference backend |
| --- | --- |
| `cpu` | CPU |
| `cuda` | NVIDIA CUDA |
| `rocm` | AMD ROCm |

CI builds and smoke-tests images on pull requests and publishes on pushes to
`main`. Commit-specific tags use `sha-<full40commit>-<backend>`. Pin an image
`@sha256:<digest>` for immutable deployment; there is no `latest` alias.

Model weights and Hugging Face tokens stay outside the images. Downloaded models
use a persistent cache. See the [deployment guide](docs/api.md) for access setup,
`docker pull`/`docker run` commands, GPU requirements, and optional source builds.

Image smoke tests do not verify full-model inference or GPU hardware support.

## Development checks

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
ruff check api_server.py api_backend.py tests
```

These tests exercise the API and backend control flow without downloading model
weights. Docker builds also check native CPU geometry and texture operations.

## Model and license

Model access is gated at
[Hugging Face](https://huggingface.co/stabilityai/stable-fast-3d).
See [LICENSE.md](LICENSE.md) for the model and code license.

[Paper](https://arxiv.org/abs/2408.00653):

```BibTeX
@article{sf3d2024,
  title={SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrapping and Illumination Disentanglement},
  author={Boss, Mark and Huang, Zixuan and Vasishta, Aaryaman and Jampani, Varun},
  journal={arXiv preprint},
  year={2024}
}
```

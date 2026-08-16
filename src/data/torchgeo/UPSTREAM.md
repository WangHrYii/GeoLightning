# TorchGeo Dataset Sources

- Upstream: https://github.com/microsoft/torchgeo
- Release: `v0.4.1`
- Commit: `15c1b255ff1aaa3b98fe2d91bf854c2bbe3d46d8`
- License: MIT, reproduced in `LICENSE`
- Imported scope: `torchgeo/datasets` and `torchgeo/samplers`

GeoLightning modifications are intentionally limited to package integration:

- `millionaid.py` uses a relative import instead of `torchgeo.datasets`.
- `__init__.py` lazily exposes public classes so optional dataset dependencies
  do not prevent unrelated datasets from being imported.
- Sampler modules use the local dataset catalog and include a Hydra-friendly
  factory.

Dataset algorithms, metadata, URLs, checksums, and class implementations remain
from the pinned upstream release.

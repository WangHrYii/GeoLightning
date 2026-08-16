# TorchVision Model Sources

- Upstream: https://github.com/pytorch/vision
- Release: `v0.28.0`
- Commit: `8fb87713a24951e639c494b0f2a8a81b5f8e33a6`
- License: BSD 3-Clause, reproduced in `LICENSE`
- Imported scope: model registry helpers and 11 classification model families

GeoLightning modifications:

- Parent-package imports target installed low-level TorchVision operators,
  transforms, logging, and download utilities.
- Model registry and model implementations remain local to this directory.
- `pyproject.toml` pins TorchVision `0.28.0` to match these source files.
- `feature_backbone.py` is a GeoLightning adapter that returns multi-scale
  feature tuples for dense prediction and regression tasks.

The source-integrated families are AlexNet, DenseNet, EfficientNet,
GoogLeNet, Inception, MaxVit, MNASNet, RegNet, ShuffleNetV2, SqueezeNet, and
VGG. Individual constructor variants are discoverable with
`list_source_models()`.

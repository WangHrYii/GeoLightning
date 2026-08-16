# TorchVision Model Sources

- Upstream: https://github.com/pytorch/vision
- Release: `v0.18.1`
- Commit: `126fc22ce33e6c2426edcf9ed540810c178fe9ce`
- License: BSD 3-Clause, reproduced in `LICENSE`
- Imported scope: model registry helpers and 11 classification model families

GeoLightning modifications:

- Parent-package imports target installed low-level TorchVision operators,
  transforms, logging, and download utilities.
- Model registry and model implementations remain local to this directory.
- `requirements.txt` pins TorchVision `0.18.1` to match these source files.
- `feature_backbone.py` is a GeoLightning adapter that returns multi-scale
  feature tuples for dense prediction and regression tasks.

The source-integrated families are AlexNet, DenseNet, EfficientNet,
GoogLeNet, Inception, MaxVit, MNASNet, RegNet, ShuffleNetV2, SqueezeNet, and
VGG. Individual constructor variants are discoverable with
`list_source_models()`.

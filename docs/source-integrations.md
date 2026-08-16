# Source Integrations

## TorchGeo datasets

TorchGeo dataset implementations are included under `src.data.torchgeo`; the
external `torchgeo` package is not imported at runtime. The catalog currently
contains 85 public dataset classes from 56 modules.

```python
from src.data import torchgeo

print(torchgeo.available_datasets())
EuroSAT = torchgeo.get_dataset_class("EuroSAT")
dataset = EuroSAT(root="data/eurosat", split="train", bands=("B04", "B03", "B02"))
```

Hydra can resolve lazy catalog classes directly:

```yaml
_target_: src.data.torchgeo.BigEarthNet
root: ${oc.env:DATA_ROOT,data}/bigearthnet
split: train
bands: all
num_classes: 19
```

For the existing Lightning training entry, use the generic non-geospatial data
module:

```yaml
_target_: src.data.TorchGeoDataModule
dataset_name: EuroSAT
root: ${oc.env:DATA_ROOT,data}/eurosat
batch_size: 64
common_kwargs:
  bands: [B04, B03, B02]
  download: false
train_kwargs: {split: train}
val_kwargs: {split: val}
test_kwargs: {split: test}
```

Core geospatial datasets require Rasterio, Fiona, PyProj, Shapely, and Rtree.
Some individual datasets raise focused import errors for optional readers such
as `h5py`, `laspy`, `pycocotools`, or `pyvista`. Radiant MLHub download support
is isolated in the `mlhub-legacy` extra because its unmaintained client requires
Pydantic 1 and Shapely 1.8, which conflict with the standard training stack.

Spatial datasets can use the integrated `random`, `random_batch`, `grid`, and
`pre_chipped` samplers. See `configs/torchgeo/chesapeake13_datamodule.yaml` for
a train/evaluation configuration using random chips and a deterministic grid.

## TorchVision source backbones

The local TorchVision source catalog contains 52 variants across AlexNet,
DenseNet, EfficientNet, GoogLeNet, Inception, MaxVit, MNASNet, RegNet,
ShuffleNetV2, SqueezeNet, and VGG.

Model definitions, constructors, weights metadata, and the registry are local
source. The installed TorchVision `0.28.0` package is used only for lower-level
operators, transforms, logging, and weight download utilities.

```python
import torch
from src.models.backbones.torchvision_source import TorchvisionSourceBackbone

encoder = TorchvisionSourceBackbone(
    model_name="efficientnet_b0",
    in_channels=6,
    pretrained=False,
)
features = encoder(torch.randn(2, 6, 256, 256))
```

The adapter returns four feature maps ordered from high to low spatial
resolution. `in_channels` can be any positive number; pretrained RGB kernels
are averaged or repeated when the first convolution is adapted.

Existing repository backbones can be normalized without changing their source:

```python
from src.models.backbones import adapt_backbone

encoder = adapt_backbone(existing_encoder)
features = encoder.forward_features(images)
print(encoder.out_channels, encoder.out_strides)
```

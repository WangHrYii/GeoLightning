"""Source-integrated TorchVision model families."""

# Importing each module registers its constructors in the local source registry.
from . import alexnet as _alexnet
from . import densenet as _densenet
from . import efficientnet as _efficientnet
from . import googlenet as _googlenet
from . import inception as _inception
from . import maxvit as _maxvit
from . import mnasnet as _mnasnet
from . import regnet as _regnet
from . import shufflenetv2 as _shufflenetv2
from . import squeezenet as _squeezenet
from . import vgg as _vgg
from ._api import get_model as create_source_model
from ._api import list_models as list_source_models
from .feature_backbone import TorchvisionSourceBackbone

__all__ = [
    "TorchvisionSourceBackbone",
    "create_source_model",
    "list_source_models",
]

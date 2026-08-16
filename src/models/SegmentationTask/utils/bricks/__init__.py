"""Compatibility layer for the canonical backbone bricks package.

New code should import from :mod:`src.models.backbones.bricks` directly. This
module preserves the small legacy builder API still used by segmentation heads.
"""

from typing import Any, Dict, Tuple

from torch import nn

from src.models.backbones.bricks import (
    FFN,
    ActivationBuilder,
    AdaptivePadding,
    AdptivePaddingConv2d,
)
from src.models.backbones.bricks import BuildActivation as _build_activation
from src.models.backbones.bricks import BuildDropout
from src.models.backbones.bricks import BuildNormalization as _build_normalization
from src.models.backbones.bricks import (
    DepthwiseSeparableConv2d,
    DropoutBuilder,
    DynamicConv2d,
    InvertedResidual,
    InvertedResidualV3,
    L2Norm,
    MultiheadAttention,
    NormalizationBuilder,
    PatchEmbed,
    PatchMerging,
    PositionEmbeddingSine,
    Scale,
    SqueezeExcitationConv2d,
    makedivisible,
    nchw2nlc2nchw,
    nchwtonlc,
    nlc2nchw2nlc,
    nlctonchw,
    truncnormal,
)

_NORMALIZATIONS = {
    "batchnorm1d": "BatchNorm1d",
    "batchnorm2d": "BatchNorm2d",
    "batchnorm3d": "BatchNorm3d",
    "groupnorm": "GroupNorm",
    "instancenorm1d": "InstanceNorm1d",
    "instancenorm2d": "InstanceNorm2d",
    "instancenorm3d": "InstanceNorm3d",
    "layernorm": "LayerNorm2d",
    "syncbatchnorm": "SyncBatchNorm",
}

_ACTIVATIONS = {
    "gelu": "GELU",
    "hardsigmoid": "HardSigmoid",
    "hardswish": "HardSwish",
    "leakyrelu": "LeakyReLU",
    "prelu": "PReLU",
    "relu": "ReLU",
    "relu6": "ReLU6",
    "sigmoid": "Sigmoid",
}


def BuildNormalization(
    norm_type: str = "batchnorm2d",
    instanced_params: Tuple[int, Dict[str, Any]] = (0, {}),
    only_get_all_supported: bool = False,
    **_: Any,
) -> nn.Module:
    """Translate the legacy normalization builder call to the canonical API."""
    if only_get_all_supported:
        return list(_NORMALIZATIONS)
    if norm_type == "identity":
        return nn.Identity()
    normalized = _NORMALIZATIONS.get(norm_type.lower(), norm_type)
    channels, options = instanced_params
    return _build_normalization(channels, {"type": normalized, **options})


def BuildActivation(activation_type: str, **kwargs: Any) -> nn.Module:
    """Translate the legacy activation builder call to the canonical API."""
    if activation_type.lower() == "identity":
        return nn.Identity()
    normalized = _ACTIVATIONS.get(activation_type.lower(), activation_type)
    return _build_activation({"type": normalized, **kwargs})

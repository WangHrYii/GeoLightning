"""Common feature-backbone protocol and compatibility adapter."""

from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from torch import Tensor, nn

FeatureTuple = Tuple[Tensor, ...]


@runtime_checkable
class FeatureBackbone(Protocol):
    """Interface consumed by dense prediction heads."""

    out_channels: Optional[Tuple[int, ...]]
    out_strides: Optional[Tuple[int, ...]]

    def forward_features(self, x: Tensor) -> FeatureTuple:
        """Return feature maps ordered from high to low spatial resolution."""


def _normalize_features(value: Any) -> FeatureTuple:
    if isinstance(value, Tensor):
        features = (value,)
    elif isinstance(value, Mapping):
        features = tuple(item for item in value.values() if isinstance(item, Tensor))
    elif isinstance(value, Sequence):
        features = tuple(item for item in value if isinstance(item, Tensor))
    else:
        raise TypeError(f"Unsupported backbone output type: {type(value).__name__}")

    if not features:
        raise ValueError("Backbone produced no tensor features")
    return features


class FeatureBackboneAdapter(nn.Module):
    """Adapt an existing module to the common feature-backbone interface."""

    def __init__(
        self,
        backbone: nn.Module,
        out_indices: Optional[Sequence[int]] = None,
        out_channels: Optional[Sequence[int]] = None,
        out_strides: Optional[Sequence[int]] = None,
        prefer_forward_features: bool = True,
        require_spatial: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.out_indices = tuple(out_indices) if out_indices is not None else None
        self.out_channels = tuple(out_channels) if out_channels is not None else None
        self.out_strides = tuple(out_strides) if out_strides is not None else None
        self.prefer_forward_features = prefer_forward_features
        self.require_spatial = require_spatial

    def _run_backbone(self, x: Tensor) -> Any:
        forward_features = getattr(self.backbone, "forward_features", None)
        if self.prefer_forward_features and callable(forward_features):
            return forward_features(x)
        return self.backbone(x)

    def forward_features(self, x: Tensor) -> FeatureTuple:
        features = _normalize_features(self._run_backbone(x))
        if self.out_indices is not None:
            try:
                features = tuple(features[index] for index in self.out_indices)
            except IndexError as exc:
                raise ValueError(
                    f"out_indices={self.out_indices} is invalid for {len(features)} features"
                ) from exc

        if self.require_spatial and any(feature.ndim != 4 for feature in features):
            shapes = [tuple(feature.shape) for feature in features]
            raise ValueError(f"Dense backbone features must be 4D tensors, got {shapes}")

        self.out_channels = tuple(int(feature.shape[1]) for feature in features)
        if all(feature.ndim == 4 for feature in features):
            height, width = x.shape[-2:]
            self.out_strides = tuple(
                max(1, round(max(height / feature.shape[-2], width / feature.shape[-1])))
                for feature in features
            )
        return features

    def forward(self, x: Tensor) -> FeatureTuple:
        return self.forward_features(x)


def adapt_backbone(backbone: nn.Module, **kwargs: Any) -> FeatureBackboneAdapter:
    """Return a common-interface wrapper for any existing backbone module."""
    if isinstance(backbone, FeatureBackboneAdapter) and not kwargs:
        return backbone
    return FeatureBackboneAdapter(backbone, **kwargs)

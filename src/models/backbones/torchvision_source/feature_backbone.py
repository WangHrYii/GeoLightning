"""Feature-pyramid adapter for source-integrated TorchVision models."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ._api import get_model

_HOOK_NODES: Dict[str, Tuple[str, ...]] = {
    "googlenet": ("conv3", "inception3b", "inception4e", "inception5b"),
    "inception_v3": ("Conv2d_4a_3x3", "Mixed_5d", "Mixed_6e", "Mixed_7c"),
}


def _replace_first_conv(module: nn.Module, in_channels: int) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if child.in_channels == in_channels:
                return

            replacement = nn.Conv2d(
                in_channels=in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups if child.groups == 1 else in_channels,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            with torch.no_grad():
                if in_channels == 1:
                    replacement.weight.copy_(child.weight.mean(dim=1, keepdim=True))
                else:
                    repeats = (in_channels + child.in_channels - 1) // child.in_channels
                    weight = child.weight.repeat(1, repeats, 1, 1)[:, :in_channels]
                    replacement.weight.copy_(weight * (child.in_channels / in_channels))
                if child.bias is not None and replacement.bias is not None:
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
            return

        try:
            _replace_first_conv(child, in_channels)
            return
        except LookupError:
            continue

    raise LookupError("No Conv2d layer found while adapting input channels")


def _append_stage(stages: List[Tensor], value: Tensor) -> None:
    if value.ndim != 4:
        return
    if stages and stages[-1].shape[-2:] == value.shape[-2:]:
        stages[-1] = value
    else:
        stages.append(value)


class TorchvisionSourceBackbone(nn.Module):
    """Turn locally integrated TorchVision classifiers into pyramid encoders.

    The returned tuple is ordered from high to low spatial resolution. By
    default, the last four distinct-resolution stages are returned.
    """

    def __init__(
        self,
        model_name: str,
        in_channels: int = 3,
        pretrained: bool = False,
        weights: Optional[Any] = None,
        out_indices: Sequence[int] = (-4, -3, -2, -1),
        freeze: bool = False,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        if pretrained and weights is None:
            weights = "DEFAULT"

        self.model_name = model_name
        self.out_indices = tuple(out_indices)
        self.model = get_model(model_name, weights=weights, **model_kwargs)
        _replace_first_conv(self.model, in_channels)

        if in_channels != 3 and hasattr(self.model, "transform_input"):
            self.model.transform_input = False

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        self.out_channels: Optional[Tuple[int, ...]] = None
        self.out_strides: Optional[Tuple[int, ...]] = None

    def _forward_sequential(self, x: Tensor, layers: nn.Module) -> List[Tensor]:
        stages: List[Tensor] = []
        for layer in layers.children():
            x = layer(x)
            _append_stage(stages, x)
        return stages

    def _forward_regnet(self, x: Tensor) -> List[Tensor]:
        x = self.model.stem(x)
        stages: List[Tensor] = [x]
        for block in self.model.trunk_output.children():
            x = block(x)
            _append_stage(stages, x)
        return stages

    def _forward_maxvit(self, x: Tensor) -> List[Tensor]:
        x = self.model.stem(x)
        stages: List[Tensor] = [x]
        for block in self.model.blocks:
            x = block(x)
            _append_stage(stages, x)
        return stages

    def _forward_shufflenet(self, x: Tensor) -> List[Tensor]:
        stages: List[Tensor] = []
        x = self.model.conv1(x)
        _append_stage(stages, x)
        x = self.model.maxpool(x)
        _append_stage(stages, x)
        for name in ("stage2", "stage3", "stage4", "conv5"):
            x = getattr(self.model, name)(x)
            _append_stage(stages, x)
        return stages

    def _forward_with_hooks(self, x: Tensor, nodes: Sequence[str]) -> List[Tensor]:
        captured: Dict[str, Tensor] = {}
        handles = []
        for name in nodes:
            module = self.model.get_submodule(name)
            handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, key=name: captured.__setitem__(key, output)
                )
            )
        try:
            self.model(x)
        finally:
            for handle in handles:
                handle.remove()
        return [captured[name] for name in nodes]

    def _forward_all_stages(self, x: Tensor) -> List[Tensor]:
        if self.model_name.startswith("regnet_"):
            return self._forward_regnet(x)
        if self.model_name.startswith("maxvit_"):
            return self._forward_maxvit(x)
        if self.model_name.startswith("shufflenet_"):
            return self._forward_shufflenet(x)
        if self.model_name in _HOOK_NODES:
            return self._forward_with_hooks(x, _HOOK_NODES[self.model_name])
        if hasattr(self.model, "features"):
            return self._forward_sequential(x, self.model.features)
        if hasattr(self.model, "layers"):
            return self._forward_sequential(x, self.model.layers)
        raise ValueError(f"Model {self.model_name!r} has no feature extraction strategy")

    def forward_features(self, x: Tensor) -> Tuple[Tensor, ...]:
        stages = self._forward_all_stages(x)
        if not stages:
            raise RuntimeError(f"Model {self.model_name!r} produced no spatial feature stages")

        try:
            outputs = tuple(stages[index] for index in self.out_indices)
        except IndexError as exc:
            raise ValueError(
                f"out_indices={self.out_indices} is invalid for {len(stages)} stages"
            ) from exc
        self.out_channels = tuple(output.shape[1] for output in outputs)
        height, width = x.shape[-2:]
        self.out_strides = tuple(
            max(1, round(max(height / output.shape[-2], width / output.shape[-1])))
            for output in outputs
        )
        return outputs

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        return self.forward_features(x)

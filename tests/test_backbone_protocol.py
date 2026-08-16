import pytest
import torch
from torch import nn

from src.models.backbones.protocol import FeatureBackboneAdapter, adapt_backbone


class TupleBackbone(nn.Module):
    def forward(self, x):
        return x[:, :2, ::2, ::2], x[:, :3, ::4, ::4]


class DictBackbone(nn.Module):
    def forward_features(self, x):
        return {"low": x[:, :1], "high": x[:, :2, ::2, ::2]}


def test_backbone_adapter_infers_metadata() -> None:
    adapter = adapt_backbone(TupleBackbone())
    outputs = adapter(torch.randn(1, 4, 32, 32))

    assert adapter.out_channels == (2, 3)
    assert adapter.out_strides == (2, 4)
    assert [output.shape[-1] for output in outputs] == [16, 8]


def test_backbone_adapter_normalizes_mapping_output() -> None:
    adapter = FeatureBackboneAdapter(DictBackbone())
    outputs = adapter.forward_features(torch.randn(1, 4, 16, 16))

    assert len(outputs) == 2
    assert adapter.out_channels == (1, 2)
    assert adapter.out_strides == (1, 2)


def test_backbone_adapter_rejects_non_spatial_features() -> None:
    adapter = FeatureBackboneAdapter(nn.Sequential(nn.Flatten()))
    with pytest.raises(ValueError, match="4D tensors"):
        adapter(torch.randn(1, 3, 8, 8))

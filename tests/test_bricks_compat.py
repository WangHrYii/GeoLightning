import torch
from torch import nn

from src.models.SegmentationTask.utils.bricks import BuildActivation, BuildNormalization


def test_legacy_bricks_delegate_to_canonical_implementations() -> None:
    normalization = BuildNormalization("layernorm", (4, {}))
    activation = BuildActivation("relu", inplace=True)

    output = activation(normalization(torch.randn(2, 4, 8, 8)))
    assert output.shape == (2, 4, 8, 8)
    assert isinstance(activation, nn.ReLU)

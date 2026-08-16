import sys

import pytest
import torch


def test_torchgeo_catalog_is_lazy_and_complete() -> None:
    from src.data import torchgeo

    datasets = torchgeo.available_datasets()
    assert len(datasets) == 85
    assert datasets == tuple(sorted(datasets))
    assert {"BigEarthNet", "EuroSAT", "LoveDA", "SEN12MS", "SpaceNet7"} <= set(datasets)
    assert "src.data.torchgeo.eurosat" not in sys.modules


def test_torchgeo_catalog_rejects_unknown_dataset() -> None:
    from src.data import torchgeo

    with pytest.raises(KeyError, match="Unknown dataset"):
        torchgeo.get_dataset_class("NotARealDataset")


def test_source_model_catalog() -> None:
    pytest.importorskip("torchvision")
    from src.models.backbones.torchvision_source import list_source_models

    models = set(list_source_models())
    assert len(models) == 52
    assert {
        "densenet121",
        "efficientnet_b0",
        "googlenet",
        "inception_v3",
        "maxvit_t",
        "regnet_x_400mf",
    } <= models


@pytest.mark.parametrize(
    "model_name",
    ["densenet121", "efficientnet_b0", "regnet_x_400mf", "shufflenet_v2_x0_5"],
)
def test_source_backbone_four_band_pyramid(model_name: str) -> None:
    pytest.importorskip("torchvision")
    from src.models.backbones.torchvision_source import TorchvisionSourceBackbone

    model = TorchvisionSourceBackbone(model_name=model_name, in_channels=4)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 4, 64, 64))

    assert model.model.__class__.__module__.startswith("src.models.backbones.torchvision_source")
    assert len(outputs) == 4
    assert model.out_channels == tuple(output.shape[1] for output in outputs)
    assert model.out_strides == tuple(64 // output.shape[-1] for output in outputs)
    spatial_shapes = [output.shape[-2:] for output in outputs]
    assert all(
        previous[0] > current[0] and previous[1] > current[1]
        for previous, current in zip(spatial_shapes, spatial_shapes[1:])
    )

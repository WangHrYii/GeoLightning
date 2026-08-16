import importlib

import pytest
import torch
from torch.utils.data import Dataset, Sampler


def test_torchgeo_non_geo_datamodule(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("src.data.TorchGeoDataModule")

    class DummyDataset(Dataset):
        def __init__(self, root: str, split: str, length: int = 5) -> None:
            self.root = root
            self.split = split
            self.length = length

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, index: int):
            return {"image": torch.tensor([index]), "split": self.split}

    monkeypatch.setattr(module, "get_dataset_class", lambda _name: DummyDataset)
    datamodule = module.TorchGeoDataModule(
        dataset_name="Dummy",
        root="data",
        batch_size=2,
        num_workers=0,
        common_kwargs={"length": 5},
    )
    datamodule.setup()

    batch = next(iter(datamodule.train_dataloader()))
    assert batch["image"].shape == (2, 1)
    assert batch["split"] == ["train", "train"]


def test_torchgeo_batch_sampler_datamodule(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("src.data.TorchGeoDataModule")

    class DummyDataset(Dataset):
        def __init__(self, root: str) -> None:
            self.root = root

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int):
            return torch.tensor(index)

    class DummyBatchSampler(Sampler):
        def __iter__(self):
            yield [0, 1]
            yield [2, 3]

        def __len__(self) -> int:
            return 2

    monkeypatch.setattr(module, "get_dataset_class", lambda _name: DummyDataset)
    monkeypatch.setattr(
        module,
        "build_geo_sampler",
        lambda _dataset, _config, _batch_size: (DummyBatchSampler(), True),
    )
    datamodule = module.TorchGeoDataModule(
        dataset_name="SpatialDummy",
        root="data",
        batch_size=8,
        num_workers=0,
        train_kwargs={},
        val_kwargs={},
        test_kwargs={},
        train_sampler={"kind": "random_batch"},
    )
    datamodule.setup("fit")

    assert next(iter(datamodule.train_dataloader())).tolist() == [0, 1]


def test_spatial_dataset_requires_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("src.data.TorchGeoDataModule")

    class GeoDataset(Dataset):
        def __init__(self, root: str, split: str) -> None:
            self.root = root

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            return index

    monkeypatch.setattr(module, "get_dataset_class", lambda _name: GeoDataset)
    datamodule = module.TorchGeoDataModule(dataset_name="SpatialDummy", root="data", num_workers=0)

    with pytest.raises(ValueError, match="requires a sampler"):
        datamodule.setup("fit")

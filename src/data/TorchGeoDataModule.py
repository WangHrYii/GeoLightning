"""Lightning adapter for source-integrated TorchGeo non-geospatial datasets."""

from typing import Any, Dict, Mapping, Optional, Tuple

import lightning as L
from torch.utils.data import DataLoader, Dataset

from src.data.torchgeo import get_dataset_class


def build_geo_sampler(
    dataset: Dataset,
    config: Mapping[str, Any],
    default_batch_size: int,
) -> Tuple[Any, bool]:
    """Load spatial sampler support only when a spatial dataset requests it."""
    from src.data.torchgeo.samplers import build_geo_sampler as build

    return build(dataset, config, default_batch_size)


class TorchGeoDataModule(L.LightningDataModule):
    """Build train, validation, and test datasets from one TorchGeo class.

    Index-based ``NonGeoDataset`` implementations use ordinary batching. Spatial
    ``GeoDataset`` implementations use local TorchGeo sampler mappings supplied
    through ``train_sampler``, ``val_sampler``, and ``test_sampler``.
    """

    def __init__(
        self,
        dataset_name: str,
        root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        common_kwargs: Optional[Mapping[str, Any]] = None,
        train_kwargs: Optional[Mapping[str, Any]] = None,
        val_kwargs: Optional[Mapping[str, Any]] = None,
        test_kwargs: Optional[Mapping[str, Any]] = None,
        train_sampler: Optional[Mapping[str, Any]] = None,
        val_sampler: Optional[Mapping[str, Any]] = None,
        test_sampler: Optional[Mapping[str, Any]] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.common_kwargs = dict(common_kwargs or {})
        self.train_kwargs = dict(train_kwargs) if train_kwargs is not None else {"split": "train"}
        self.val_kwargs = dict(val_kwargs) if val_kwargs is not None else {"split": "val"}
        self.test_kwargs = dict(test_kwargs) if test_kwargs is not None else {"split": "test"}
        self.train_sampler_config = dict(train_sampler) if train_sampler is not None else None
        self.val_sampler_config = dict(val_sampler) if val_sampler is not None else None
        self.test_sampler_config = dict(test_sampler) if test_sampler is not None else None
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.drop_last = drop_last
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self._train_sampler: Optional[Any] = None
        self._val_sampler: Optional[Any] = None
        self._test_sampler: Optional[Any] = None
        self._train_batch_sampler = False
        self._val_batch_sampler = False
        self._test_batch_sampler = False

    def _build_dataset(self, split_kwargs: Mapping[str, Any]) -> Dataset:
        dataset_class = get_dataset_class(self.dataset_name)
        kwargs: Dict[str, Any] = {"root": self.root, **self.common_kwargs, **split_kwargs}
        dataset = dataset_class(**kwargs)
        if not isinstance(dataset, Dataset):
            raise TypeError(f"{self.dataset_name} did not construct a torch Dataset")
        return dataset

    def _build_sampler(
        self, dataset: Dataset, config: Optional[Mapping[str, Any]]
    ) -> Tuple[Optional[Any], bool]:
        if config is None:
            if any(base.__name__ == "GeoDataset" for base in type(dataset).__mro__):
                raise ValueError(
                    f"{self.dataset_name} is a GeoDataset and requires a sampler config"
                )
            return None, False
        return build_geo_sampler(dataset, config, self.batch_size)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate"):
            self.data_train = self._build_dataset(self.train_kwargs)
            self.data_val = self._build_dataset(self.val_kwargs)
            self._train_sampler, self._train_batch_sampler = self._build_sampler(
                self.data_train, self.train_sampler_config
            )
            self._val_sampler, self._val_batch_sampler = self._build_sampler(
                self.data_val, self.val_sampler_config
            )
        if stage in (None, "test", "predict"):
            self.data_test = self._build_dataset(self.test_kwargs)
            self._test_sampler, self._test_batch_sampler = self._build_sampler(
                self.data_test, self.test_sampler_config
            )

    def _loader(
        self,
        dataset: Optional[Dataset],
        sampler: Optional[Any],
        batch_sampler: bool,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        if dataset is None:
            raise RuntimeError("setup() must be called before requesting a dataloader")
        common = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if batch_sampler:
            return DataLoader(dataset, batch_sampler=sampler, **common)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            drop_last=drop_last,
            **common,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(
            self.data_train,
            self._train_sampler,
            self._train_batch_sampler,
            shuffle=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader(
            self.data_val,
            self._val_sampler,
            self._val_batch_sampler,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self._loader(
            self.data_test,
            self._test_sampler,
            self._test_batch_sampler,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

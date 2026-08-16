"""Configuration factory for source-integrated TorchGeo samplers."""

from typing import Any, Mapping, Tuple

from torch.utils.data import Dataset, Sampler

from .. import BoundingBox
from .batch import BatchGeoSampler, RandomBatchGeoSampler
from .constants import Units
from .single import GridGeoSampler, PreChippedGeoSampler, RandomGeoSampler

_SAMPLERS = {
    "grid": GridGeoSampler,
    "pre_chipped": PreChippedGeoSampler,
    "random": RandomGeoSampler,
    "random_batch": RandomBatchGeoSampler,
}


def build_geo_sampler(
    dataset: Dataset,
    config: Mapping[str, Any],
    default_batch_size: int,
) -> Tuple[Sampler, bool]:
    """Build a local TorchGeo sampler from a Hydra-friendly mapping.

    Returns the sampler and whether it is a batch sampler.
    """
    kwargs = dict(config)
    kind = str(kwargs.pop("kind", "")).lower()
    if kind not in _SAMPLERS:
        choices = ", ".join(sorted(_SAMPLERS))
        raise ValueError(f"Unknown geo sampler {kind!r}. Available samplers: {choices}")

    units = kwargs.get("units")
    if isinstance(units, str):
        try:
            kwargs["units"] = Units[units.upper()]
        except KeyError as exc:
            raise ValueError("Sampler units must be 'pixels' or 'crs'") from exc

    roi = kwargs.get("roi")
    if isinstance(roi, (list, tuple)):
        if len(roi) != 6:
            raise ValueError("Sampler roi must contain six bounding-box values")
        kwargs["roi"] = BoundingBox(*roi)

    sampler_class = _SAMPLERS[kind]
    if sampler_class is RandomBatchGeoSampler:
        kwargs.setdefault("batch_size", default_batch_size)

    sampler = sampler_class(dataset, **kwargs)
    return sampler, isinstance(sampler, BatchGeoSampler)

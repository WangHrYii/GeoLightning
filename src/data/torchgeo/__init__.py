# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Modified for GeoLightning: public classes are loaded lazily.

"""Source-integrated TorchGeo datasets.

The implementation is vendored from TorchGeo v0.4.1. Importing this package is
dependency-light; individual dataset modules load their geospatial or optional
dependencies only when the corresponding class is requested.
"""

from importlib import import_module
from typing import Any, Dict, Tuple

_PUBLIC_SYMBOLS: Dict[str, str] = {
    "ADVANCE": "advance",
    "AbovegroundLiveWoodyBiomassDensity": "agb_live_woody_density",
    "AsterGDEM": "astergdem",
    "BeninSmallHolderCashews": "benin_cashews",
    "BigEarthNet": "bigearthnet",
    "BoundingBox": "utils",
    "CDL": "cdl",
    "CMSGlobalMangroveCanopy": "cms_mangrove_canopy",
    "COWC": "cowc",
    "COWCCounting": "cowc",
    "COWCDetection": "cowc",
    "CV4AKenyaCropType": "cv4a_kenya_crop_type",
    "CanadianBuildingFootprints": "cbf",
    "Chesapeake": "chesapeake",
    "Chesapeake7": "chesapeake",
    "Chesapeake13": "chesapeake",
    "ChesapeakeCVPR": "chesapeake",
    "ChesapeakeDC": "chesapeake",
    "ChesapeakeDE": "chesapeake",
    "ChesapeakeMD": "chesapeake",
    "ChesapeakeNY": "chesapeake",
    "ChesapeakePA": "chesapeake",
    "ChesapeakeVA": "chesapeake",
    "ChesapeakeWV": "chesapeake",
    "CloudCoverDetection": "cloud_cover",
    "DFC2022": "dfc2022",
    "DeepGlobeLandCover": "deepglobelandcover",
    "EDDMapS": "eddmaps",
    "ETCI2021": "etci2021",
    "EUDEM": "eudem",
    "EnviroAtlas": "enviroatlas",
    "Esri2020": "esri2020",
    "EuroSAT": "eurosat",
    "EuroSAT100": "eurosat",
    "FAIR1M": "fair1m",
    "ForestDamage": "forestdamage",
    "GBIF": "gbif",
    "GID15": "gid15",
    "GeoDataset": "geo",
    "GlobBiomass": "globbiomass",
    "IDTReeS": "idtrees",
    "INaturalist": "inaturalist",
    "InriaAerialImageLabeling": "inria",
    "IntersectionDataset": "geo",
    "LEVIRCDPlus": "levircd",
    "LandCoverAI": "landcoverai",
    "Landsat": "landsat",
    "Landsat1": "landsat",
    "Landsat2": "landsat",
    "Landsat3": "landsat",
    "Landsat4MSS": "landsat",
    "Landsat4TM": "landsat",
    "Landsat5MSS": "landsat",
    "Landsat5TM": "landsat",
    "Landsat7": "landsat",
    "Landsat8": "landsat",
    "Landsat9": "landsat",
    "LoveDA": "loveda",
    "MillionAID": "millionaid",
    "NAIP": "naip",
    "NASAMarineDebris": "nasa_marine_debris",
    "NonGeoClassificationDataset": "geo",
    "NonGeoDataset": "geo",
    "OSCD": "oscd",
    "OpenBuildings": "openbuildings",
    "PatternNet": "patternnet",
    "Potsdam2D": "potsdam",
    "RESISC45": "resisc45",
    "RasterDataset": "geo",
    "ReforesTree": "reforestree",
    "SEN12MS": "sen12ms",
    "SeasonalContrastS2": "seco",
    "Sentinel": "sentinel",
    "Sentinel1": "sentinel",
    "Sentinel2": "sentinel",
    "So2Sat": "so2sat",
    "SpaceNet": "spacenet",
    "SpaceNet1": "spacenet",
    "SpaceNet2": "spacenet",
    "SpaceNet3": "spacenet",
    "SpaceNet4": "spacenet",
    "SpaceNet5": "spacenet",
    "SpaceNet6": "spacenet",
    "SpaceNet7": "spacenet",
    "TropicalCyclone": "cyclone",
    "UCMerced": "ucmerced",
    "USAVars": "usavars",
    "UnionDataset": "geo",
    "VHR10": "vhr10",
    "Vaihingen2D": "vaihingen",
    "VectorDataset": "geo",
    "XView2": "xview",
    "ZueriCrop": "zuericrop",
    "concat_samples": "utils",
    "merge_samples": "utils",
    "stack_samples": "utils",
    "unbind_samples": "utils",
}

_NON_DATASET_SYMBOLS = {
    "BoundingBox",
    "GeoDataset",
    "IntersectionDataset",
    "NonGeoClassificationDataset",
    "NonGeoDataset",
    "RasterDataset",
    "UnionDataset",
    "VectorDataset",
    "concat_samples",
    "merge_samples",
    "stack_samples",
    "unbind_samples",
}


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC_SYMBOLS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{_PUBLIC_SYMBOLS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def available_datasets() -> Tuple[str, ...]:
    """Return dataset class names without importing their modules."""
    return tuple(sorted(set(_PUBLIC_SYMBOLS) - _NON_DATASET_SYMBOLS))


def get_dataset_class(name: str) -> Any:
    """Load a dataset class by public TorchGeo name."""
    if name not in available_datasets():
        available = ", ".join(available_datasets())
        raise KeyError(f"Unknown dataset {name!r}. Available datasets: {available}")
    return getattr(__import__(__name__, fromlist=[name]), name)


__all__ = tuple(_PUBLIC_SYMBOLS) + ("available_datasets", "get_dataset_class")

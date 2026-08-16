# Dependency Profiles

- `core.txt`: configuration, tabular data, and common utilities.
- `train.txt`: tested PyTorch 2.13.0 / TorchVision 0.28.0 training stack.
- `geo.txt`: geospatial readers and spatial indexing.
- `datasets.txt`: optional readers required by individual TorchGeo datasets.
- `dev.txt`: full development and verification environment.

Radiant MLHub downloads use the separate `mlhub-legacy` extra because the
unmaintained client requires Pydantic 1 and Shapely 1.8. Install it in a
dedicated environment; it is incompatible with the standard `train,geo` stack.

`pyproject.toml` is the single source of truth for dependency versions. The
profile files only select matching extras, which prevents duplicate dependency
graph and Dependabot entries.

For a CUDA install, install the matching PyTorch wheels first, then use the
profile files. CPU CI uses the official PyTorch CPU wheel index.

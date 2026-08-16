# Dependency Profiles

- `core.txt`: configuration, tabular data, and common utilities.
- `train.txt`: tested PyTorch 2.3.1 / TorchVision 0.18.1 training stack.
- `geo.txt`: geospatial readers and spatial indexing.
- `datasets.txt`: optional readers required by individual TorchGeo datasets.
- `dev.txt`: full development and verification environment.
- `constraints-py38.txt`: tested Python 3.8 and CUDA 11.8 direct dependency pins.

For a CUDA install, install the matching PyTorch wheels first, then use the
profile files. CPU CI uses the official PyTorch CPU wheel index.

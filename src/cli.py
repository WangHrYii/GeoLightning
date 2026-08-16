"""Console entry points for GeoLightning."""

from importlib import import_module
from typing import NoReturn


def _run(module_name: str) -> NoReturn:
    module = import_module(module_name)
    module.main()
    raise SystemExit(0)


def train_cli() -> NoReturn:
    """Run the default training application."""
    return _run("src.train")


def eval_cli() -> NoReturn:
    """Run the evaluation application."""
    return _run("src.eval")


def inference_cli() -> NoReturn:
    """Run the inference application."""
    return _run("src.inference")

import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/TreeHeight_DPT/config.yaml",
        "configs/AutoMAE/config.yaml",
        "configs/RSIPAC_25_T1/MCANet/config.yaml",
        "configs/torchgeo/eurosat_datamodule.yaml",
        "configs/backbones/torchvision_source.yaml",
    ],
)
def test_primary_configs_parse(project_root: Path, relative_path: str) -> None:
    config = OmegaConf.load(project_root / relative_path)
    assert config


def test_production_configs_are_machine_independent(project_root: Path) -> None:
    forbidden = ("/home/", "/mnt/", "/root/", "/Users/")
    offenders = []
    paths = list((project_root / "configs").rglob("*.yaml"))
    paths.extend(
        path
        for path in (project_root / "src").rglob("*.py")
        if "src/data/torchgeo" not in path.as_posix()
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if any(prefix in text for prefix in forbidden):
            offenders.append(str(path.relative_to(project_root)))
    assert offenders == []


def test_security_audit_export_contains_runtime_pins(project_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "audit.txt"
    subprocess.run(
        [sys.executable, "tools/export_audit_requirements.py", str(output)],
        cwd=project_root,
        check=True,
    )
    requirements = output.read_text(encoding="utf-8").splitlines()
    assert "torch==2.13.0" in requirements
    assert "torchvision==0.28.0" in requirements
    assert all("github.com" not in requirement for requirement in requirements)

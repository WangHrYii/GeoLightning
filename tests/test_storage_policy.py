import importlib.util
import os
from datetime import datetime, timezone
from pathlib import Path


def load_prune_module(project_root: Path):
    path = project_root / "tools/prune_outputs.py"
    spec = importlib.util.spec_from_file_location("prune_outputs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_output_retention_keeps_latest_and_protected(project_root: Path, tmp_path: Path) -> None:
    module = load_prune_module(project_root)
    runs = []
    for index in range(4):
        run = tmp_path / f"run-{index}"
        (run / ".hydra").mkdir(parents=True)
        timestamp = 1_700_000_000 + index
        os.utime(run, (timestamp, timestamp))
        runs.append(run)
    (runs[0] / ".keep").touch()

    ordered = module.discover_runs(tmp_path)
    candidates = module.plan_prune(
        ordered,
        keep_latest=1,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert runs[3] not in candidates
    assert runs[0] not in candidates
    assert candidates == [runs[2], runs[1]]

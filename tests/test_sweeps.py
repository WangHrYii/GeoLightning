from pathlib import Path


def test_hydra_run_directories_are_ignored(project_root: Path) -> None:
    gitignore = (project_root / ".gitignore").read_text(encoding="utf-8")
    assert "/outputs/" in gitignore
    assert "/logs/" in gitignore

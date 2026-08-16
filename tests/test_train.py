import subprocess
import sys
from pathlib import Path


def test_train_entrypoint_help(project_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, "src/train.py", "--help"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Configuration" in result.stdout


def test_console_entry_module_imports() -> None:
    from src import cli

    assert callable(cli.train_cli)
    assert callable(cli.eval_cli)
    assert callable(cli.inference_cli)

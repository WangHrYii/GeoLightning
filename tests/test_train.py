import subprocess
import sys
from pathlib import Path

import pytest
import torch
from lightning import LightningModule


class CheckpointModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)


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


def test_untrusted_checkpoint_instantiator_is_blocked(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "untrusted.ckpt"
    torch.save(
        {
            "state_dict": CheckpointModel().state_dict(),
            "hyper_parameters": {"_instantiator": "os.system"},
            "pytorch-lightning_version": "2.6.2",
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="blocked to prevent arbitrary code execution"):
        CheckpointModel.load_from_checkpoint(checkpoint_path)

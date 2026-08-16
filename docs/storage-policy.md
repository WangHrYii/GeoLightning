# Storage Policy

Every maintained training config keeps at most one monitored best checkpoint
plus `last.ckpt`. Full checkpoint state is retained where training resume is a
supported workflow; feature-upsampling configs keep weights-only checkpoints.

Hydra run directories are retained independently. Preview the current policy:

```bash
python tools/prune_outputs.py outputs --keep-latest 20 --max-age-days 90
```

Add an empty `.keep` file inside an important run directory to protect it. The
tool only deletes data when `--apply` is explicitly supplied.

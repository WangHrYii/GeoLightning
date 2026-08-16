#!/usr/bin/env python3
"""Plan or apply retention rules to Hydra output directories."""

import argparse
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional


def discover_runs(root: Path) -> List[Path]:
    """Return Hydra run directories ordered from newest to oldest."""
    if not root.exists():
        return []
    runs = {hydra_dir.parent for hydra_dir in root.rglob(".hydra") if hydra_dir.is_dir()}
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def plan_prune(
    runs: Iterable[Path],
    keep_latest: int,
    max_age_days: Optional[int] = None,
    now: Optional[datetime] = None,
) -> List[Path]:
    """Select unprotected runs outside the configured retention window."""
    if keep_latest < 0:
        raise ValueError("keep_latest must be non-negative")
    ordered = list(runs)
    cutoff = None
    if max_age_days is not None:
        if max_age_days < 0:
            raise ValueError("max_age_days must be non-negative")
        current = now or datetime.now(timezone.utc)
        cutoff = current - timedelta(days=max_age_days)

    candidates = []
    retained_unprotected = 0
    for run in ordered:
        if (run / ".keep").exists():
            continue
        if retained_unprotected < keep_latest:
            retained_unprotected += 1
            continue
        if cutoff is not None:
            modified = datetime.fromtimestamp(run.stat().st_mtime, timezone.utc)
            if modified >= cutoff:
                continue
        candidates.append(run)
    return candidates


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("outputs"))
    parser.add_argument("--keep-latest", type=int, default=20)
    parser.add_argument("--max-age-days", type=int)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete selected runs. Without this flag the command is a dry run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = discover_runs(args.root)
    candidates = plan_prune(runs, args.keep_latest, args.max_age_days)
    total = sum(directory_size(path) for path in candidates)

    action = "DELETE" if args.apply else "WOULD DELETE"
    for path in candidates:
        print(f"{action} {path} ({human_size(directory_size(path))})")
        if args.apply:
            shutil.rmtree(path)

    mode = "applied" if args.apply else "dry run"
    print(
        f"Retention {mode}: {len(candidates)} of {len(runs)} runs, "
        f"{human_size(total)} selected."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

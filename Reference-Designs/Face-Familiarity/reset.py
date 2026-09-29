#!/usr/bin/env python3.12
"""
Wipe the CURRENT dataset's operational data: its library.db, thumbnails, and
score reports (see dataset.py — data/<dataset>/). Other datasets are never
touched — resetting "employees" leaves "30-rock" (or any other dataset)
completely alone. Source files (common.py, db.py, ingest.py, score.py,
server.py, static/, CLAUDE.md, ...) are untouched.

Downloaded videos and cached EyePop results (downloads/, cache/) are shared
across ALL datasets and are NOT cleared by default — wiping them would force
every OTHER dataset to re-download/re-run EyePop too, not just this one. Pass
--include-shared-cache if you specifically need to invalidate those (e.g.
after a Pop/model change — see common.py:POP_FINGERPRINT for a lighter-weight
alternative that only invalidates the results cache, not downloads).

Usage:
    python3.12 reset.py                        # resets the current dataset only
    python3.12 reset.py --yes                  # skip the confirmation prompt
    python3.12 reset.py --include-shared-cache # also wipes downloads/ and cache/
"""
import argparse
import shutil
from pathlib import Path

import dataset

SHARED_TARGETS = [Path("downloads"), Path("cache")]


def _dataset_targets() -> list[Path]:
    d = dataset.DATA_ROOT / dataset.get_current()
    return [d / "library.db", d / "thumbnails", d / "reports"]


def wipe(targets: list[Path]) -> None:
    for target in targets:
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        print(f"removed {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="skip confirmation prompt")
    parser.add_argument(
        "--include-shared-cache", action="store_true",
        help="also wipe downloads/ and cache/ — shared across ALL datasets, not just this one",
    )
    args = parser.parse_args()

    targets = _dataset_targets() + (SHARED_TARGETS if args.include_shared_cache else [])
    existing = [t for t in targets if t.exists()]
    if not existing:
        print(f"Nothing to clean for dataset '{dataset.get_current()}' — already fresh.")
        raise SystemExit(0)

    print(f"This will permanently delete (dataset: '{dataset.get_current()}'):")
    for t in existing:
        print(f"  {t}")
    if args.include_shared_cache:
        print("(--include-shared-cache: downloads/ and cache/ are SHARED across every dataset, not just this one)")
    if args.yes or input("Proceed? [y/N] ").strip().lower() == "y":
        wipe(existing)
        print("Done.")
    else:
        print("Aborted.")

#!/usr/bin/env python3.12
"""
Wipe all Face-Familiarity operational data: the face library (library.db),
thumbnails, downloaded videos, cached frames, and score reports. Source files
(common.py, db.py, ingest.py, score.py, server.py, static/, CLAUDE.md, ...)
are untouched.

Usage:
    python3.12 reset.py         # prompts for confirmation
    python3.12 reset.py --yes   # skip the prompt
"""
import argparse
import shutil
from pathlib import Path

TARGETS = [
    Path("library.db"),
    Path("thumbnails"),
    Path("downloads"),
    Path("cache"),
    Path("reports"),
]


def wipe() -> None:
    for target in TARGETS:
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
    args = parser.parse_args()

    existing = [t for t in TARGETS if t.exists()]
    if not existing:
        print("Nothing to clean — already fresh.")
        raise SystemExit(0)

    print("This will permanently delete:")
    for t in existing:
        print(f"  {t}")
    if args.yes or input("Proceed? [y/N] ").strip().lower() == "y":
        wipe()
        print("Done. Operational data cleared.")
    else:
        print("Aborted.")

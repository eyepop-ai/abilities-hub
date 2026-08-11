#!/usr/bin/env python3.12
"""
Per-dataset storage for Face-Familiarity — lets the same code operate on
separate face libraries (e.g. one video series vs. another) without mixing
labeling work. Each dataset gets its own library.db/thumbnails/reports under
data/<slug>/. Switch with set_current() — the label UI's dataset switcher
calls this through server.py's /api/datasets routes; scripts can set the
DATASET env var instead. The current selection is persisted to
data/.current so it survives a server restart — DATASET, if set, overrides
that persisted choice (e.g. for scripting/CI use).

Deliberately NOT dataset-scoped: common.py's DOWNLOAD_DIR/FRAME_CACHE_DIR/
RESULTS_CACHE_DIR. Downloaded videos and cached EyePop results are keyed by
video id/stem already, so sharing them across datasets is safe and avoids
redundant downloads/EyePop calls if a video is ever reused across datasets.

Process-wide global, not per-request/session — fine for this tool's
single-user local use; switching datasets from two browser tabs at once would
race. See CLAUDE.md.
"""
import os
import re
from pathlib import Path

DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True)
_CURRENT_FILE = DATA_ROOT / ".current"

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError("dataset name must contain at least one letter or digit")
    return slug


def _dataset_dir(name: str) -> Path:
    return DATA_ROOT / name


def list_datasets() -> list[str]:
    return sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())


def set_current(name: str) -> str:
    """Switch the active dataset, creating it (and its subdirs) if new, and
    persisting the choice to data/.current so it survives a server restart.
    Returns the resolved slug."""
    global _current
    slug = slugify(name)
    _current = slug
    thumbnail_dir()  # side effect: ensures data/<slug>/{thumbnails,reports} exist
    report_dir()
    _CURRENT_FILE.write_text(slug)
    return slug


def get_current() -> str:
    return _current


def db_path() -> Path:
    return _dataset_dir(_current) / "library.db"


def thumbnail_dir() -> Path:
    d = _dataset_dir(_current) / "thumbnails"
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_dir() -> Path:
    d = _dataset_dir(_current) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _initial_dataset() -> str:
    if os.getenv("DATASET"):
        return os.environ["DATASET"]
    if _CURRENT_FILE.exists():
        return _CURRENT_FILE.read_text().strip() or "default"
    return "default"


_current = "default"
set_current(_initial_dataset())

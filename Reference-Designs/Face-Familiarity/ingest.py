#!/usr/bin/env python3.12
"""
Ingest a video into the face library.

Samples the video every SAMPLE_INTERVAL_SECONDS (or --interval), runs each
frame through the Person -> Face -> Face Embedding Pop (see common.py), and
stores one unlabeled face row (thumbnail + embedding) per detection. Faces sit
unassigned until a human groups and names them in the label UI (see server.py
/ static/label.html).

Every run appends per-frame timing (video-decode time vs. EyePop round-trip
time) and a summary to logs/ingest.log, so ingest performance can be compared
across runs (e.g. resolution or model changes) — see CLAUDE.md.

Usage:
    python3.12 ingest.py <youtube_url_or_path> [--interval 0.5]
"""
import argparse
import asyncio
import time
from datetime import datetime
from pathlib import Path

import dataset
import db
from common import crop_jpeg, embedding_to_bytes, extract_faces, is_good_face_shot, resolve_video, scan_video, thumbnail_dir

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "ingest.log"


def _log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


async def ingest_video(source: str, progress=None, interval: float | None = None) -> int:
    run_start = time.monotonic()
    video = resolve_video(source)
    _log(
        f"=== ingest start: {video.name} (source={source}, dataset={dataset.get_current()}, "
        f"interval={interval if interval is not None else 'default'}) ==="
    )

    conn = db.connect()
    count = 0
    skipped = 0
    grab_times: list[float] = []
    eyepop_times: list[float] = []

    def timing(seconds, grab_ms, eyepop_ms):
        grab_times.append(grab_ms)
        eyepop_times.append(eyepop_ms)
        _log(f"  frame @ {seconds:7.2f}s  grab={grab_ms:6.0f}ms  eyepop={eyepop_ms:6.0f}ms")

    async for seconds, jpeg, result in scan_video(video, progress, timing, interval=interval):
        for face in extract_faces(result):
            if not is_good_face_shot(face):
                skipped += 1
                continue
            thumb_name = f"{video.stem}_{seconds:.2f}s_{db.new_uuid()[:8]}.jpg"
            (thumbnail_dir() / thumb_name).write_bytes(crop_jpeg(jpeg, face["face_bbox"]))
            db.insert_face(
                conn,
                source=video.name,
                seconds=seconds,
                confidence=face["confidence"],
                thumbnail_path=thumb_name,
                embedding_blob=embedding_to_bytes(face["embedding"]),
            )
            count += 1
    conn.close()

    elapsed_s = time.monotonic() - run_start
    total_eyepop_s = sum(eyepop_times) / 1000
    total_grab_s = sum(grab_times) / 1000
    avg_eyepop_ms = (total_eyepop_s * 1000 / len(eyepop_times)) if eyepop_times else 0
    eyepop_share = (total_eyepop_s / elapsed_s * 100) if elapsed_s else 0
    _log(
        f"=== ingest done: {video.name} -- {count} face(s) ingested, {skipped} skipped "
        f"(no eyes visible or too small) from {len(eyepop_times)} frame(s) in "
        f"{elapsed_s:.1f}s wall time ({total_eyepop_s:.1f}s in EyePop calls, {eyepop_share:.0f}%; "
        f"{total_grab_s:.1f}s decoding frames; avg {avg_eyepop_ms:.0f}ms/frame in EyePop) ==="
    )
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="YouTube URL or local video path")
    parser.add_argument("--interval", type=float, default=None,
                         help="seconds between sampled frames (default: SAMPLE_INTERVAL_SECONDS, "
                              "usually 2 — use a smaller value like 0.25-0.5 for short clips)")
    args = parser.parse_args()
    ingested = asyncio.run(
        ingest_video(args.source, progress=lambda i, n: print(f"  frame {i}/{n}", end="\r"), interval=args.interval)
    )
    print(f"\nIngested {ingested} face(s) into the library. Open the label UI to group and name them.")

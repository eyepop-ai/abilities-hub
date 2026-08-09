#!/usr/bin/env python3.12
"""
Score a video for face familiarity against the labeled library.

For every face detected, finds the closest labeled person by nearest-neighbor
cosine similarity — the best match against any ONE of that person's individual
stored faces, not a centroid/average. A centroid gets washed out once someone
has many faces spanning different looks (glasses on/off, angles, lighting) —
confirmed on real data: a threshold that gave 80% recall / 0% false-match-rate
with a centroid gives 94% recall / 0.2% false-match-rate with nearest-neighbor
at the exact same threshold. See CLAUDE.md. Below `threshold`, the face is
reported as "Unknown" rather than forced into the nearest label.

Usage:
    python3.12 score.py <youtube_url_or_path> [--threshold 0.45]

Writes face thumbnails to ./thumbnails and a JSON report to
./reports/<video_stem>.json for the review UI (see server.py / static/review.html).
"""
import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import db
from common import FAMILIARITY_THRESHOLD, THUMBNAIL_DIR, bytes_to_embedding, crop_jpeg, extract_faces, is_good_face_shot, normalize_rows, resolve_video, scan_video

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def load_person_embeddings(conn) -> dict[str, tuple[str, np.ndarray]]:
    """Every stored embedding per labeled person, as one L2-normalized matrix —
    normalized once here (not per detected face) so match_person's per-person
    nearest-neighbor lookup is a single dot product against a matrix, not a
    Python loop over individual vectors. See CLAUDE.md (same fix as
    server.py:_person_suggestions, same measured ~250x speedup)."""
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    names: dict[str, str] = {}
    for row in db.labeled_faces(conn):
        grouped[row["person_uuid"]].append(bytes_to_embedding(row["embedding"]))
        names[row["person_uuid"]] = row["name"]
    return {
        person_uuid: (names[person_uuid], normalize_rows(np.stack(vectors)))
        for person_uuid, vectors in grouped.items()
    }


def match_person(embedding: np.ndarray, people: dict[str, tuple[str, np.ndarray]], threshold: float):
    """Nearest-neighbor match: best similarity against any single stored face per
    person, not an average — see module docstring for why."""
    norm = np.linalg.norm(embedding)
    query = embedding / norm if norm else embedding
    best_uuid, best_name, best_score = None, "Unknown", -1.0
    for person_uuid, (name, matrix) in people.items():
        score = float((matrix @ query).max())
        if score > best_score:
            best_uuid, best_name, best_score = person_uuid, name, score
    if best_uuid is None or best_score < threshold:
        return None, "Unknown", round(best_score, 4)
    return best_uuid, best_name, round(best_score, 4)


async def score_video(source: str, threshold: float = FAMILIARITY_THRESHOLD, progress=None, on_update=None) -> dict:
    """`on_update(report_snapshot)` is called with a fresh copy of the report-so-
    far — once immediately after the video resolves (before any frame is
    scanned, so a caller can start streaming/playing the video right away) and
    then again after every processed frame, so the review UI can show the video
    playing while detections/overlay boxes fill in progressively rather than
    waiting for the whole scan to finish. video_path is valid from the first
    call; source_width/height are None until the first frame's Pop result
    arrives."""
    video = resolve_video(source)
    conn = db.connect()
    people = load_person_embeddings(conn)

    detections: list[dict] = []
    report = {
        "source": source,
        "video": video.name,
        "video_stem": video.stem,
        "video_path": str(video),
        "source_width": None,
        "source_height": None,
        "threshold": threshold,
        "generated_at": time.time(),
        "detections": detections,
    }
    if on_update:
        on_update({**report, "detections": []})

    async for seconds, jpeg, result in scan_video(video, progress):
        if report["source_width"] is None:
            report["source_width"] = result.get("source_width")
            report["source_height"] = result.get("source_height")
        for face in extract_faces(result):
            if not is_good_face_shot(face):
                continue
            person_uuid, name, similarity = match_person(face["embedding"], people, threshold)
            thumb_name = f"score_{video.stem}_{seconds:.2f}s_{db.new_uuid()[:8]}.jpg"
            (THUMBNAIL_DIR / thumb_name).write_bytes(crop_jpeg(jpeg, face["face_bbox"]))
            detections.append({
                "uuid": db.new_uuid(),
                "seconds": seconds,
                "confidence": round(face["confidence"], 4),
                "thumbnail": thumb_name,
                "embedding": face["embedding"].tolist(),
                "person_bbox": list(face["person_bbox"]),
                "face_bbox": list(face["face_bbox"]),
                "matched_person_uuid": person_uuid,
                "matched_name": name,
                "similarity": similarity,
                "familiar": person_uuid is not None,
            })
        if on_update:
            on_update({**report, "detections": list(detections)})
    conn.close()

    (REPORT_DIR / f"{video.stem}.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="YouTube URL or local video path")
    parser.add_argument("--threshold", type=float, default=FAMILIARITY_THRESHOLD)
    args = parser.parse_args()
    report = asyncio.run(
        score_video(args.source, args.threshold, progress=lambda i, n: print(f"  frame {i}/{n}", end="\r"))
    )
    known = sum(1 for d in report["detections"] if d["familiar"])
    print(
        f"\n{len(report['detections'])} face(s) detected, {known} familiar, "
        f"{len(report['detections']) - known} unknown."
    )

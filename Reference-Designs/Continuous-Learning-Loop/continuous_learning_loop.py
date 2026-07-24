#!/usr/bin/env python3.12
"""
Continuous Learning Loop — post-process, ingest, autolabel.

Pipeline (see CLAUDE.md):
  1. Run every video in PROD_VIDEO_DIR through the original trained model, with
     tracking enabled so each detection carries a stable trackId.
  2. Select candidate frames: low-confidence detections, plus "flicker" frames —
     a track that disappears for a frame or two and then reappears, which catches
     near-miss false negatives a flat confidence threshold would miss.
  3. Cap samples per video and enforce a minimum time gap between samples from
     the same video, so one noisy clip can't dominate the retrain set.
  4. Upload the selected frames into the original dataset (EyePop versions it
     automatically) and kick off auto-label.
  5. Print the dashboard URL for human review.

Usage:
    python3.12 continuous_learning_loop.py [--dry-run]

`--dry-run` runs inference and candidate selection but skips uploading and
auto-label, so you can sanity-check what would be sampled first.

Requires videos in ./production_videos (override with PROD_VIDEO_DIR env var).
"""
import argparse
import asyncio
import io
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
from dotenv import load_dotenv
from eyepop import EyePopSdk
from eyepop.worker.worker_types import (
    CropForward, InferenceComponent, MotionModel, Pop, TrackingComponent,
)

load_dotenv(Path(__file__).parent / ".env", override=True)

API_KEY = os.environ["EYEPOP_API_KEY"]
ACCOUNT_UUID = os.environ["ACCOUNT_UUID"]
ORIGINAL_DATASET_UUID = os.environ["ORIGINAL_DATASET_UUID"]
ORIGINAL_MODEL_UUID = os.environ["ORIGINAL_MODEL_UUID"]

PROD_VIDEO_DIR = Path(os.getenv("PROD_VIDEO_DIR", "production_videos"))
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

CONFIDENCE_THRESHOLD = float(os.getenv("LOOP_CONFIDENCE_THRESHOLD", "0.6"))
MAX_SAMPLES_PER_VIDEO = int(os.getenv("LOOP_MAX_SAMPLES_PER_VIDEO", "5"))
MIN_GAP_SECONDS = float(os.getenv("LOOP_MIN_GAP_SECONDS", "10"))
TRACK_MAX_AGE_SECONDS = float(os.getenv("LOOP_TRACK_MAX_AGE_SECONDS", "2"))
MAX_FLICKER_GAP_FRAMES = int(os.getenv("LOOP_MAX_FLICKER_GAP_FRAMES", "1"))

# Deliberately much lower than CONFIDENCE_THRESHOLD: this is the server-side
# cutoff EyePop applies before returning any detections at all. If it matched
# CONFIDENCE_THRESHOLD, everything in the 0.2-0.6 band we want to flag as
# "low confidence" would be prefiltered away before we ever see it.
MIN_INFERENCE_CONFIDENCE = float(os.getenv("LOOP_MIN_INFERENCE_CONFIDENCE", "0.2"))

POP = Pop(components=[
    InferenceComponent(
        abilityUuid=ORIGINAL_MODEL_UUID,
        confidenceThreshold=MIN_INFERENCE_CONFIDENCE,
        forward=CropForward(
            targets=[
                TrackingComponent(
                    maxAgeSeconds=TRACK_MAX_AGE_SECONDS,
                    motionModel=MotionModel.CONSTANT_VELOCITY,
                    agnostic=True,
                )
            ]
        ),
    )
])


async def run_inference(endpoint, video: Path) -> list[dict]:
    """Process one video at native frame rate (tracking needs continuous frames)."""
    cache = CACHE_DIR / f"{video.stem}.jsonl"
    if cache.exists():
        return [json.loads(line) for line in cache.read_text().splitlines()]
    frames = []
    with open(cache, "w") as f:
        job = await endpoint.upload(str(video))
        while result := await job.predict():
            frames.append(result)
            f.write(json.dumps(result) + "\n")
    return frames


def select_candidates(frames: list[dict], conf_threshold: float, max_gap_frames: int) -> list[dict]:
    """Flag low-confidence detections and flicker frames (track drops out, then reappears)."""
    track_history = defaultdict(list)  # trackId -> [(frame_index, seconds, confidence)]
    low_confidence = []

    for i, frame in enumerate(frames):
        seconds = frame.get("seconds", float(i))
        for obj in frame.get("objects", []):
            confidence = obj.get("confidence")
            track_id = obj.get("trackId")
            if confidence is None:
                continue
            if track_id is not None:
                track_history[track_id].append((i, seconds, confidence))
            if confidence < conf_threshold:
                low_confidence.append({
                    "frame_index": i, "seconds": seconds,
                    "reason": "low_confidence", "confidence": confidence,
                })

    flicker = []
    for history in track_history.values():
        history.sort(key=lambda h: h[0])
        for (prev_idx, prev_sec, prev_conf), (next_idx, next_sec, next_conf) in zip(history, history[1:]):
            if next_idx - prev_idx > max_gap_frames + 1:
                flicker.append({
                    "frame_index": prev_idx, "seconds": prev_sec,
                    "reason": "flicker_gap", "confidence": prev_conf,
                })
                flicker.append({
                    "frame_index": next_idx, "seconds": next_sec,
                    "reason": "flicker_gap", "confidence": next_conf,
                })

    return flicker + low_confidence


def dedupe_and_cap(candidates: list[dict], min_gap_seconds: float, max_samples: int) -> list[dict]:
    """Prefer flicker candidates, then lowest confidence; drop anything within min_gap_seconds of a pick."""
    ranked = sorted(candidates, key=lambda c: (c["reason"] != "flicker_gap", c["confidence"]))
    selected = []
    for candidate in ranked:
        if any(abs(candidate["seconds"] - s["seconds"]) < min_gap_seconds for s in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_samples:
            break
    return selected


def extract_frame_jpeg(video: Path, seconds: float) -> bytes | None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else None


def ingest_samples(data, video: Path, samples: list[dict]) -> int:
    ingested = 0
    for sample in samples:
        jpeg = extract_frame_jpeg(video, sample["seconds"])
        if jpeg is None:
            print(f"  ! could not extract frame at {sample['seconds']:.2f}s from {video.name}, skipping")
            continue
        external_id = f"{video.stem}_{sample['seconds']:.2f}s_{sample['reason']}"
        job = data.upload_asset_job(
            io.BytesIO(jpeg), mime_type="image/jpeg",
            dataset_uuid=ORIGINAL_DATASET_UUID, external_id=external_id,
        )
        asset = job.result()
        while True:
            asset = data.get_asset(asset.uuid, dataset_uuid=ORIGINAL_DATASET_UUID)
            if asset.status == "accepted":
                break
            time.sleep(1)
        print(f"  + ingested {external_id} (conf={sample['confidence']:.2f})")
        ingested += 1
    return ingested


async def main(dry_run: bool):
    if not PROD_VIDEO_DIR.exists():
        raise SystemExit(
            f"No production video group found at {PROD_VIDEO_DIR}. "
            f"Add production videos there, or set PROD_VIDEO_DIR."
        )
    video_extensions = ("*.mp4", "*.mov", "*.MOV", "*.MP4")
    videos = sorted({p for ext in video_extensions for p in PROD_VIDEO_DIR.glob(ext)})
    if not videos:
        raise SystemExit(f"No video files found in {PROD_VIDEO_DIR}.")

    per_video_samples: list[tuple[Path, list[dict]]] = []

    print("Connecting to EyePop...")
    async with EyePopSdk.async_worker(api_key=API_KEY) as endpoint:
        print("Connected.")
        await endpoint.set_pop(POP)
        for video in videos:
            print(f"Processing {video.name}...")
            frames = await run_inference(endpoint, video)
            candidates = select_candidates(frames, CONFIDENCE_THRESHOLD, MAX_FLICKER_GAP_FRAMES)
            samples = dedupe_and_cap(candidates, MIN_GAP_SECONDS, MAX_SAMPLES_PER_VIDEO)
            print(f"  {len(frames)} frames -> {len(candidates)} candidates -> {len(samples)} selected")
            per_video_samples.append((video, samples))

    total_selected = sum(len(s) for _, s in per_video_samples)
    if total_selected == 0:
        print("No candidates met the sampling criteria. Nothing to ingest.")
        return

    if dry_run:
        preview_dir = Path("dry_run_preview")
        preview_dir.mkdir(exist_ok=True)
        print(f"\n--dry-run: would ingest {total_selected} sample(s):")
        for video, samples in per_video_samples:
            for s in samples:
                print(f"  {video.name} @ {s['seconds']:.2f}s ({s['reason']}, conf={s['confidence']:.2f})")
                jpeg = extract_frame_jpeg(video, s["seconds"])
                if jpeg is None:
                    print(f"    ! could not extract frame at {s['seconds']:.2f}s, skipping preview")
                    continue
                out_path = preview_dir / f"{video.stem}_{s['seconds']:.2f}s_{s['reason']}.jpg"
                out_path.write_bytes(jpeg)
        print(f"\nPreview frames written to {preview_dir}/")
        return

    print(f"\nIngesting {total_selected} sample(s) into dataset {ORIGINAL_DATASET_UUID}...")
    with EyePopSdk.dataEndpoint(api_key=API_KEY, account_id=ACCOUNT_UUID) as data:
        ingested = 0
        for video, samples in per_video_samples:
            ingested += ingest_samples(data, video, samples)
        if ingested == 0:
            print("No frames were successfully ingested; skipping auto-label.")
            return
        print("Kicking off auto-label...")
        data.auto_annotate_dataset_version(dataset_uuid=ORIGINAL_DATASET_UUID)

    review_url = (
        "https://dashboard.eyepop.ai/wizardModel?type=object&step=autoLabel"
        f"&accountUUID={ACCOUNT_UUID}&modelUUID={ORIGINAL_MODEL_UUID}&datasetUUID={ORIGINAL_DATASET_UUID}"
    )
    print(f"\nReview and train here:\n{review_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="select candidates without ingesting or auto-labeling")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))

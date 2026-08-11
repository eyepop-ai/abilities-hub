#!/usr/bin/env python3.12
"""
Shared config, EyePop Pop definition, video/frame helpers, and embedding math
for the Face-Familiarity reference design (see CLAUDE.md).

Used by ingest.py, score.py, and server.py.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from eyepop import EyePopSdk
from eyepop.worker.worker_types import CropForward, InferenceComponent, Pop

from dataset import report_dir, thumbnail_dir  # re-exported: most callers already `from common import ...`

load_dotenv(Path(__file__).parent / ".env", override=True)

API_KEY = os.environ["EYEPOP_API_KEY"]
EYEPOP_URL = os.getenv("EYEPOP_URL", "https://compute.eyepop.ai")

# /watch (see server.py, static/watch.html) needs the browser to talk to
# EyePop directly over WebRTC, but the raw API_KEY must never reach client-side
# JS — anyone who can read the page source could then use it themselves outside
# this app entirely. mint_eyepop_token() exchanges it server-side for a
# short-lived bearer token via the SAME endpoint the JS/Python SDKs use
# internally for API-key auth (confirmed by reading eyepop's own SDK source):
# POST {EYEPOP_URL}/v1/auth/authenticate with Authorization: Bearer <API_KEY>,
# no body. Only that token is ever sent to the browser (server.py:
# /api/eyepop-token); the browser then connects with
# EyePopSdk.EyePop.workerEndpoint({accessToken: token, ...}) instead of apiKey.
# Observed expires_in ~78,000s (~22h) on staging — comfortably longer than any
# real /watch session, so this caches the token and re-mints only once actually
# close to expiry, and the client never needs its own refresh logic.
_token_cache = {"access_token": None, "expires_at": 0.0}


def mint_eyepop_token() -> str:
    now = time.time()
    if _token_cache["access_token"] and _token_cache["expires_at"] > now + 300:
        return _token_cache["access_token"]
    req = urllib.request.Request(
        f"{EYEPOP_URL}/v1/auth/authenticate",
        headers={"Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = now + data["expires_in"]
    return _token_cache["access_token"]


SAMPLE_INTERVAL_SECONDS = float(os.getenv("SAMPLE_INTERVAL_SECONDS", "2"))
FAMILIARITY_THRESHOLD = float(os.getenv("FAMILIARITY_THRESHOLD", "0.45"))

# Shared across all datasets (not per-dataset) — see dataset.py's module
# docstring for why: these are content-addressed by video id/stem, not by
# which face library you're currently building.
DOWNLOAD_DIR = Path("downloads")
UPLOAD_DIR = Path("uploads")
FRAME_CACHE_DIR = Path("cache/frames")
RESULTS_CACHE_DIR = Path("cache/results")
for _dir in (DOWNLOAD_DIR, UPLOAD_DIR, FRAME_CACHE_DIR, RESULTS_CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Bump this whenever POP's abilities change — it's part of the results-cache key
# (see scan_video) so a model swap can't silently serve stale cached results.
POP_FINGERPRINT = "person-face_short_range-faceid_large"

POP = Pop(components=[
    InferenceComponent(
        ability="eyepop.person:latest",
        categoryName="person",
        forward=CropForward(
            maxItems=128,
            targets=[InferenceComponent(
                ability="eyepop.person.face.short-range:latest",
                categoryName="2d-face-points",
                forward=CropForward(
                    boxPadding=1.5,
                    orientationTargetAngle=-90.0,
                    targets=[InferenceComponent(ability="eyepop.face-id.large:latest")],
                ),
            )],
        ),
    )
])


def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _youtube_id(url: str) -> str | None:
    """Best-effort video id extraction from common YouTube URL shapes, with no
    network call — used to check the download cache before ever talking to
    yt-dlp. Returns None for anything not obviously a standard YouTube URL;
    callers should fall back to letting yt-dlp resolve it."""
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/") or None
    if parsed.hostname and "youtube" in parsed.hostname:
        return parse_qs(parsed.query).get("v", [None])[0]
    return None


def download_youtube(url: str) -> Path:
    """Download a YouTube video to DOWNLOAD_DIR, reusing an existing download by video id.

    Checks the cache path before calling yt-dlp at all — not just relying on
    yt-dlp's own skip-if-exists behavior — so a re-scan of an already-downloaded
    video needs zero network calls, not even the metadata lookup.
    """
    video_id = _youtube_id(url)
    if video_id:
        cached = DOWNLOAD_DIR / f"{video_id}.mp4"
        if cached.exists():
            return cached

    import yt_dlp

    with yt_dlp.YoutubeDL({
        # bestvideo/bestaudio first, capped at 1080p — a bare "mp4" alternative up front
        # would grab a low-res progressive stream (YouTube's combined mp4 tops out low)
        # before ever trying the higher-res adaptive streams.
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "outtmpl": str(DOWNLOAD_DIR / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "noplaylist": True,
        # Without a JS runtime, yt-dlp can't solve YouTube's token challenge and
        # extraction 403s. Deno is yt-dlp's default runtime and isn't installed
        # here; node is already present (nvm + Homebrew) so use that instead.
        "js_runtimes": {"node": {}},
    }) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
        if not path.exists():
            # merge_output_format didn't kick in (e.g. already a single mp4 stream)
            path = Path(ydl.prepare_filename(info))
        return path


def resolve_video(source: str) -> Path:
    """Return a local video path, downloading it first if `source` is a URL."""
    if not is_url(source):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"No such video file: {source}")
        return path
    return download_youtube(source)


def save_uploaded_file(filename: str, data: bytes) -> Path:
    """Save a dropped/picked file (image OR video — same handling either way,
    see below) to UPLOAD_DIR, content-addressed by hash so re-uploading the
    same file is a no-op cache hit (same idea as download_youtube's video-id
    cache). This is the cross-platform replacement for a native file picker:
    a browser <input type="file"> or drag-and-drop drop deliberately never
    exposes a real filesystem path (security), only the file's bytes and
    name, so ingest_video() needs an actual on-disk path to hand to
    cv2.VideoCapture — this function is what creates one. Images need no
    separate code path downstream: cv2.VideoCapture reads a still image as a
    single-frame capture (fps=25, frame_count=1), so ingest_video() just
    treats it as a very short video."""
    import hashlib

    digest = hashlib.sha1(data).hexdigest()[:16]
    suffix = Path(filename).suffix.lower() or ".jpg"
    path = UPLOAD_DIR / f"{digest}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    return path


def _open_video(video: Path) -> cv2.VideoCapture:
    """cv2.VideoCapture with the container's rotation metadata applied. Phone
    videos (e.g. iPhone portrait recordings) are stored in landscape sensor
    orientation with a rotation flag for players to apply — cv2 does NOT apply
    it by default, so every frame comes out sideways (confirmed on real iPhone
    footage: 1920x1080 landscape frames from a video that's actually 1080x1920
    portrait). CAP_PROP_ORIENTATION_AUTO (OpenCV 4.5+) fixes this at the source
    for every reader, rather than needing a manual rotation step downstream."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


def sample_frame_seconds(video: Path, interval: float) -> list[float]:
    """Evenly spaced timestamps covering the video's duration at `interval` seconds apart."""
    cap = _open_video(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    duration = frame_count / fps if fps else 0.0
    steps = max(1, int(duration // interval) + 1)
    return [round(i * interval, 2) for i in range(steps)]


def grab_frame_jpeg(video: Path, seconds: float) -> bytes | None:
    cap = _open_video(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else None


def crop_jpeg(jpeg: bytes, bbox: tuple[float, float, float, float]) -> bytes:
    """Crop a JPEG to `bbox` (x, y, width, height); falls back to the full frame on failure."""
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    x, y, w, h = bbox
    x, y = max(0, int(x)), max(0, int(y))
    crop = arr[y:y + max(1, int(h)), x:x + max(1, int(w))]
    if crop.size == 0:
        crop = arr
    ok, buf = cv2.imencode(".jpg", crop)
    return buf.tobytes() if ok else jpeg


def _extract_embedding(face_obj: dict) -> np.ndarray | None:
    for raw in face_obj.get("raw", []):
        for tensor in raw.get("tensors", []):
            if tensor.get("name") != "embedding":
                continue
            data = tensor["data"]
            flat = data[0] if data and isinstance(data[0], list) else data
            return np.array(flat, dtype=np.float32)
    return None


def _eye_landmarks(face_obj: dict) -> tuple[bool, float]:
    """(has_both_eyes, landmark_confidence). The face's keyPoints landmarks are
    all-or-nothing in practice (measured across 3,749 real detections: every
    face has either all 6 landmarks — both eyes, nose, mouth, both ears — or
    none at all, never a partial set), so checking for both eyes is equivalent
    to checking landmarks are present at all. But presence alone is NOT a
    reliable quality signal: below ~0.5 confidence the model omits points
    entirely rather than emitting low-confidence guesses, yet points emitted
    just above that floor are frequently garbage — visually confirmed on real
    examples a user flagged as "no eyes visible" that nonetheless had all 6
    labels present at confidence ~0.54. A follow-up manual sample (9 faces
    across 0.50-0.60/0.60-0.70/0.70-0.80 confidence bands) found the 0.50-0.60
    band was garbage 3/3 times (hand, black frame, shoulder — no face at all),
    0.60-0.70 was a coin flip, and 0.70-0.80 was legitimate profile faces 3/3
    times — hence MIN_LANDMARK_CONFIDENCE. Confidence is one value for the whole
    keyPoints group, not exposed per-landmark."""
    kps = face_obj.get("keyPoints", [])
    labels = {p.get("classLabel") for kp in kps for p in kp.get("points", [])}
    has_both_eyes = "left eye" in labels and "right eye" in labels
    confidence = kps[0]["confidence"] if kps else 0.0
    return has_both_eyes, confidence


# Below this, min(width, height) puts a face in the smallest ~4% seen across two
# real videos (p5 ~85px) — likely a distant/background face, not a usable shot
# for identity matching. Face size and eye-visibility are uncorrelated (measured:
# ~95.5-96.6% both-eyes-visible at every size bucket, including the smallest), so
# this is a genuinely separate filter, not redundant with the eye checks.
MIN_FACE_SIZE_PX = 80

# See _eye_landmarks — the 0.50 emission floor is not a quality bar, this is.
MIN_LANDMARK_CONFIDENCE = 0.65


def is_good_face_shot(face: dict) -> bool:
    """True if a face dict from extract_faces() is a usable shot for identity
    matching: both eyes visible with real landmark confidence, and not too
    small. See CLAUDE.md for the data behind these checks."""
    if not face["has_both_eyes"] or face["landmark_confidence"] < MIN_LANDMARK_CONFIDENCE:
        return False
    _, _, w, h = face["face_bbox"]
    return min(w, h) >= MIN_FACE_SIZE_PX


def extract_faces(result: dict) -> list[dict]:
    """Flatten a Pop result into one dict per detected face with bboxes + embedding.

    Drops every face in a frame if 2+ of them share a byte-identical embedding.
    Confirmed EyePop worker bug (reproduced directly against the raw API, not a
    parsing issue on our side): a frame with multiple people can have every face's
    "embedding" tensor come back identical — apparently the first face's result
    gets attached to every face slot instead of each getting its own. We can't
    tell which one (if any) is genuinely correct, so trusting any of them risks a
    confident wrong match; dropping them all is the safe default. See CLAUDE.md.
    """
    faces = []
    for person in result.get("objects", []):
        if person.get("classLabel") != "person":
            continue
        for obj in person.get("objects", []):
            if obj.get("classLabel") != "face":
                continue
            embedding = _extract_embedding(obj)
            if embedding is None:
                continue
            has_both_eyes, landmark_confidence = _eye_landmarks(obj)
            faces.append({
                "person_bbox": (person["x"], person["y"], person["width"], person["height"]),
                "face_bbox": (obj["x"], obj["y"], obj["width"], obj["height"]),
                "confidence": obj.get("confidence", 0.0),
                "embedding": embedding,
                "has_both_eyes": has_both_eyes,
                "landmark_confidence": landmark_confidence,
            })

    if len(faces) < 2:
        return faces
    blobs = [f["embedding"].tobytes() for f in faces]
    dupe_blobs = {b for b in blobs if blobs.count(b) > 1}
    if not dupe_blobs:
        return faces
    return [f for f, b in zip(faces, blobs) if b not in dupe_blobs]


def _results_cache_path(video: Path, interval: float) -> Path:
    return RESULTS_CACHE_DIR / f"{video.stem}_{interval}s_{POP_FINGERPRINT}.jsonl"


def _load_cached_results(cache_path: Path) -> dict[float, dict]:
    if not cache_path.exists():
        return {}
    cached = {}
    for line in cache_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cached[row["seconds"]] = row["result"]
    return cached


async def scan_video(video: Path, progress=None, timing=None, interval: float | None = None):
    """Sample `video` every `interval` seconds (default SAMPLE_INTERVAL_SECONDS —
    2s is too sparse for short clips, e.g. a 10s phone video would get ~5
    samples total; pass a smaller interval for those) and run each frame
    through POP.

    Yields (seconds, frame_jpeg, pop_result) for every sampled frame that decoded
    successfully. `progress(done, total)` is called after each frame if given.
    `timing(seconds, grab_ms, eyepop_ms)` is called after each frame's EyePop
    round trip if given (0 for cache hits), to measure wall-clock vs. EyePop-side
    latency.

    EyePop results are cached to RESULTS_CACHE_DIR keyed by video + interval +
    POP_FINGERPRINT (see that constant) — re-scanning the same video at the SAME
    interval needs no EyePop connection at all if every sampled frame is already
    cached; a different interval gets its own cache file rather than reusing or
    invalidating the other one, since the two sample sets don't overlap cleanly.
    Frame jpegs are always re-grabbed locally (cheap, no network) since cropping
    thumbnails needs the actual bytes, not just the cached inference result.
    """
    interval = SAMPLE_INTERVAL_SECONDS if interval is None else interval
    seconds_list = sample_frame_seconds(video, interval)
    cache_path = _results_cache_path(video, interval)
    cached_results = _load_cached_results(cache_path)
    missing = [s for s in seconds_list if s not in cached_results]

    if missing:
        print("Connecting to EyePop...")
        async with EyePopSdk.async_worker(api_key=API_KEY) as endpoint:
            print("Connected.")
            await endpoint.set_pop(POP)
            with open(cache_path, "a") as cache_file:
                async for item in _scan_frames(video, seconds_list, cached_results, endpoint, cache_file, progress, timing):
                    yield item
    else:
        print(f"Using {len(seconds_list)} cached EyePop result(s) for {video.name} — no EyePop connection needed.")
        async for item in _scan_frames(video, seconds_list, cached_results, None, None, progress, timing):
            yield item


async def _scan_frames(video, seconds_list, cached_results, endpoint, cache_file, progress, timing):
    for i, seconds in enumerate(seconds_list):
        grab_start = time.monotonic()
        jpeg = grab_frame_jpeg(video, seconds)
        grab_ms = (time.monotonic() - grab_start) * 1000
        if jpeg is not None:
            if seconds in cached_results:
                result = cached_results[seconds]
                eyepop_ms = 0.0
            else:
                frame_path = FRAME_CACHE_DIR / f"{video.stem}_{seconds:.2f}.jpg"
                frame_path.write_bytes(jpeg)
                eyepop_start = time.monotonic()
                job = await endpoint.upload(str(frame_path))
                result = await job.predict()
                eyepop_ms = (time.monotonic() - eyepop_start) * 1000
                cache_file.write(json.dumps({"seconds": seconds, "result": result}) + "\n")
                cache_file.flush()
            if timing:
                timing(seconds, grab_ms, eyepop_ms)
        if progress:
            progress(i + 1, len(seconds_list))
        if jpeg is not None:
            yield seconds, jpeg, result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0
    #return float(np.sum(a * b))


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row so a plain dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between every row of `a` and every row of `b`,
    shape (len(a), len(b)) — one matmul instead of a Python double loop calling
    cosine_similarity() per pair. Same exact result, ~250x faster at real
    library scale (measured: 809x464 embeddings, 1170ms loop vs 4.5ms matmul) —
    not an approximation, just not doing redundant norm recomputation and
    Python-level call overhead per pair. See CLAUDE.md."""
    return normalize_rows(a) @ normalize_rows(b).T


def embedding_to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def bytes_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

#!/usr/bin/env python3.12
"""
Face-Familiarity local UI.

  /        group & label the face library
  /review  scan a new video and score it against the library

Usage: python3.12 server.py  ->  open http://localhost:8080
"""
import asyncio
import base64
import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

import dataset
import db
import ingest
import score
from common import EYEPOP_URL, FAMILIARITY_THRESHOLD, bytes_to_embedding, embedding_to_bytes, mint_eyepop_token, save_uploaded_file, similarity_matrix, thumbnail_dir

PORT = int(os.getenv("PORT", "8080"))
STATIC_DIR = Path(__file__).parent / "static"

_VIDEO_CONTENT_TYPES = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
}

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _start_job(work) -> str:
    """Run `work(progress, set_partial) -> result` in a background thread; returns
    a pollable job id. `set_partial(data)` lets long-running work (score_video)
    publish an in-progress snapshot the frontend can read via /api/jobs/<id>
    while `status` is still "running" — e.g. so a video can start playing before
    the whole scan finishes."""
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "progress": None, "result": None, "error": None, "partial": None}

    def progress(done, total):
        with JOBS_LOCK:
            JOBS[job_id]["progress"] = {"done": done, "total": total}

    def set_partial(data):
        with JOBS_LOCK:
            JOBS[job_id]["partial"] = data

    def run():
        try:
            result = asyncio.run(work(progress, set_partial))
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "done"
                JOBS[job_id]["result"] = result
        except Exception as e:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = str(e)

    threading.Thread(target=run, daemon=True).start()
    return job_id


def _similar_faces(face_uuid: str, limit: int = 60) -> list[dict]:
    conn = db.connect()
    target = db.get_face(conn, face_uuid)
    if target is None:
        return []
    target_vec = bytes_to_embedding(target["embedding"])
    rows = conn.execute("SELECT * FROM faces WHERE uuid != ?", (face_uuid,)).fetchall()
    conn.close()
    if not rows:
        return []

    candidates = np.stack([bytes_to_embedding(r["embedding"]) for r in rows])
    scores = similarity_matrix(target_vec[None, :], candidates)[0]

    scored = [
        {"uuid": r["uuid"], "thumbnail_path": r["thumbnail_path"], "person_uuid": r["person_uuid"], "similarity": round(float(s), 4)}
        for r, s in zip(rows, scores)
    ]
    scored.sort(key=lambda s: -s["similarity"])
    return scored[:limit]


def _face_to_json(row) -> dict:
    return {k: row[k] for k in row.keys() if k != "embedding"}


def _people_embeddings() -> list[dict]:
    """Every labeled person in the current dataset with their raw embeddings, for
    /watch's client-side nearest-neighbor matching (mirrors score.py's
    load_person_embeddings, but JSON — the browser has no numpy, just a plain JS
    loop over these, pre-normalized once client-side on load)."""
    conn = db.connect()
    grouped: dict[str, dict] = {}
    for row in db.labeled_faces(conn):
        person = grouped.setdefault(row["person_uuid"], {"uuid": row["person_uuid"], "name": row["name"], "embeddings": []})
        person["embeddings"].append(bytes_to_embedding(row["embedding"]).tolist())
    conn.close()
    return list(grouped.values())


def _resolve_exact_duplicates() -> int:
    """Auto-assign every unassigned face whose embedding is an exact bit-for-bit
    duplicate of an already-labeled face to that same person. Not a similarity
    guess — a same-frame duplicate detection carries an identical embedding to its
    sibling box, so this is unambiguous. See CLAUDE.md."""
    conn = db.connect()
    blob_to_person = {
        r["embedding"]: r["person_uuid"]
        for r in conn.execute("SELECT embedding, person_uuid FROM faces WHERE person_uuid IS NOT NULL")
    }
    unassigned = db.list_faces(conn, "unassigned")
    pairs = [(blob_to_person[r["embedding"]], r["uuid"]) for r in unassigned if r["embedding"] in blob_to_person]
    db.assign_face_pairs(conn, pairs)
    conn.close()
    return len(pairs)


def _person_suggestions(person_uuid: str, limit: int = 60) -> list[dict]:
    """Unassigned faces ranked by cosine similarity to whichever of this person's
    stored faces they match best (nearest-neighbor), most similar first.

    Not a centroid: once a person has many faces spanning different angles/
    expressions, a single mean/sum prototype gets washed out — a candidate only
    needs to be close to ONE of the person's stored faces, not close to their
    average. Also excludes unassigned faces that are exact-duplicate embeddings
    of an already-labeled face — same-frame duplicate detections (the person or
    face detector firing more than one box on one face) add no new information
    and otherwise dominate the ranking; see CLAUDE.md.
    """
    conn = db.connect()
    person_rows = db.list_faces(conn, person_uuid)
    if not person_rows:
        conn.close()
        return []
    labeled_blobs = {
        r["embedding"] for r in conn.execute("SELECT embedding FROM faces WHERE person_uuid IS NOT NULL")
    }
    unassigned = [r for r in db.list_faces(conn, "unassigned") if r["embedding"] not in labeled_blobs]
    conn.close()
    if not unassigned:
        return []

    person_matrix = np.stack([bytes_to_embedding(r["embedding"]) for r in person_rows])
    unassigned_matrix = np.stack([bytes_to_embedding(r["embedding"]) for r in unassigned])
    best_scores = similarity_matrix(unassigned_matrix, person_matrix).max(axis=1)

    scored = [
        {"uuid": r["uuid"], "thumbnail_path": r["thumbnail_path"], "similarity": round(float(s), 4)}
        for r, s in zip(unassigned, best_scores)
    ]
    scored.sort(key=lambda s: -s["similarity"])
    return scored[:limit]


def _cluster_unassigned(threshold: float, min_size: int, limit: int = 40) -> dict:
    """Group unassigned faces by cosine similarity so a person can be assigned to
    many faces at once instead of hunting one "similar face" at a time.

    Complete-linkage, not star: a candidate only joins if it's above `threshold`
    against EVERY face already in the cluster, not just the seed. Star clustering
    (candidate vs. seed only) lets one ambiguous face bridge two different people
    into one group — complete-linkage trades some recall for that not happening.
    """
    conn = db.connect()
    rows = db.list_faces(conn, "unassigned")
    conn.close()
    if not rows:
        return {"clusters": [], "dropped_clusters": 0, "singleton_count": 0}

    embeddings = np.stack([bytes_to_embedding(r["embedding"]) for r in rows])
    sim = similarity_matrix(embeddings, embeddings)

    order = sorted(range(len(rows)), key=lambda i: -rows[i]["confidence"])
    assigned = np.zeros(len(rows), dtype=bool)
    clusters = []
    for seed in order:
        if assigned[seed]:
            continue
        remaining = [i for i in range(len(rows)) if not assigned[i] and i != seed]
        remaining.sort(key=lambda i: -sim[seed, i])
        members = [seed]
        for c in remaining:
            if sim[seed, c] < threshold:
                break  # sorted by similarity to seed descending — nothing further qualifies
            if sim[c, members].min() >= threshold:
                members.append(c)
        if len(members) < min_size:
            continue
        assigned[members] = True
        clusters.append({
            "representative_uuid": rows[seed]["uuid"],
            "representative_thumbnail": rows[seed]["thumbnail_path"],
            "size": len(members),
            "members": [
                {"uuid": rows[i]["uuid"], "thumbnail_path": rows[i]["thumbnail_path"]}
                for i in members
            ],
        })

    clusters.sort(key=lambda c: -c["size"])
    grouped_count = sum(c["size"] for c in clusters)
    return {
        "clusters": clusters[:limit],
        "dropped_clusters": max(0, len(clusters) - limit),
        "singleton_count": len(rows) - grouped_count,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    # -- helpers -----------------------------------------------------
    def _respond(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, code: int, payload):
        self._respond(code, "application/json", json.dumps(payload).encode())

    def _serve_file(self, path: Path, content_type: str):
        if not path.exists():
            self._respond(404, "text/plain", b"Not found")
            return
        self._respond(200, content_type, path.read_bytes())

    def _serve_watch(self):
        """Template-injects the EyePop URL/default threshold into watch.html at
        serve time, rather than shipping them in the static file on disk. The
        API_KEY itself is deliberately NOT injected here — see
        common.py:mint_eyepop_token and GET /api/eyepop-token below; the
        browser gets only a short-lived bearer token, never the long-lived key."""
        html = (STATIC_DIR / "watch.html").read_text()
        html = html.replace("__EYEPOP_URL__", json.dumps(EYEPOP_URL))
        html = html.replace("__FAMILIARITY_THRESHOLD__", json.dumps(FAMILIARITY_THRESHOLD))
        self._respond(200, "text/html", html.encode())

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw) if raw else {}

    def _serve_video(self, video_path: Path):
        """Streams a video with Range support so the <video> element can seek.

        `video_path` is whatever score.py resolved and read to run inference — same
        trust boundary as running score.py with that path directly from the CLI.
        """
        if not video_path.exists() or video_path.suffix.lower() not in _VIDEO_CONTENT_TYPES:
            self._respond(404, "text/plain", b"Video not found")
            return
        content_type = _VIDEO_CONTENT_TYPES[video_path.suffix.lower()]
        file_size = video_path.stat().st_size
        range_header = self.headers.get("Range")

        if range_header:
            m = re.match(r"bytes=(\d+)-(\d*)", range_header)
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            try:
                with open(video_path, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except BrokenPipeError:
                pass
        else:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            try:
                with open(video_path, "rb") as f:
                    while chunk := f.read(65536):
                        self.wfile.write(chunk)
            except BrokenPipeError:
                pass

    # -- GET -----------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)

        if path == "/":
            self._serve_file(STATIC_DIR / "label.html", "text/html")
        elif path == "/review":
            self._serve_file(STATIC_DIR / "review.html", "text/html")
        elif path == "/watch":
            self._serve_watch()
        elif path == "/eyepop.min.js":
            self._serve_file(STATIC_DIR / "eyepop.min.js", "application/javascript")
        elif path == "/api/people/embeddings":
            self._json(200, _people_embeddings())
        elif path == "/api/eyepop-token":
            self._json(200, {"access_token": mint_eyepop_token()})
        elif path.startswith("/thumbnail/"):
            name = unquote(path[len("/thumbnail/"):])
            thumb_path = (thumbnail_dir() / name).resolve()
            if thumbnail_dir().resolve() not in thumb_path.parents:
                self._respond(403, "text/plain", b"Forbidden")
                return
            self._serve_file(thumb_path, "image/jpeg")
        elif path.startswith("/video/"):
            self._serve_video(Path(unquote(path[len("/video/"):])))
        elif path == "/api/datasets":
            self._json(200, {"current": dataset.get_current(), "available": dataset.list_datasets()})
        elif path == "/api/people":
            conn = db.connect()
            people = [dict(r) for r in db.list_people(conn)]
            conn.close()
            self._json(200, people)
        elif path == "/api/faces":
            person_uuid = query.get("person", ["unassigned"])[0]
            conn = db.connect()
            rows = db.list_faces(conn, None if person_uuid == "all" else person_uuid)
            conn.close()
            self._json(200, [_face_to_json(r) for r in rows])
        elif path.startswith("/api/faces/") and path.endswith("/similar"):
            face_uuid = path.split("/")[3]
            self._json(200, _similar_faces(face_uuid))
        elif path == "/api/faces/clusters":
            threshold = float(query.get("threshold", ["0.40"])[0])
            min_size = int(query.get("min_size", ["3"])[0])
            self._json(200, _cluster_unassigned(threshold, min_size))
        elif path.startswith("/api/people/") and path.endswith("/suggestions"):
            person_uuid = path.split("/")[3]
            self._json(200, _person_suggestions(person_uuid))
        elif path.startswith("/api/jobs/"):
            job_id = path.split("/")[3]
            job = JOBS.get(job_id)
            self._json(200, job) if job else self._json(404, {"error": "unknown job"})
        else:
            self._respond(404, "text/plain", b"Not found")

    # -- POST ------------------------------------------------------------
    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_json_body()

        if path == "/api/datasets":
            slug = dataset.set_current(body["name"])
            self._json(200, {"current": slug, "available": dataset.list_datasets()})
        elif path == "/api/upload-file":
            path_ = save_uploaded_file(body["filename"], base64.b64decode(body["data"]))
            self._json(200, {"path": str(path_)})
        elif path == "/api/people":
            conn = db.connect()
            person_uuid = db.create_person(conn, body["name"])
            conn.close()
            self._json(200, {"uuid": person_uuid})
        elif path.startswith("/api/people/") and path.endswith("/rename"):
            person_uuid = path.split("/")[3]
            conn = db.connect()
            db.rename_person(conn, person_uuid, body["name"])
            conn.close()
            self._json(200, {"ok": True})
        elif path.startswith("/api/people/") and path.endswith("/delete"):
            person_uuid = path.split("/")[3]
            conn = db.connect()
            db.delete_person(conn, person_uuid)
            conn.close()
            self._json(200, {"ok": True})
        elif path == "/api/faces/assign":
            conn = db.connect()
            db.assign_faces(conn, body["face_uuids"], body.get("person_uuid"))
            conn.close()
            self._json(200, {"ok": True})
        elif path == "/api/faces/resolve-duplicates":
            self._json(200, {"resolved": _resolve_exact_duplicates()})
        elif path == "/api/ingest":
            source = body["source"]
            interval = body.get("interval")
            job_id = _start_job(lambda progress, set_partial: ingest.ingest_video(source, progress, interval=interval))
            self._json(200, {"job_id": job_id})
        elif path == "/api/score":
            source = body["source"]
            threshold = body.get("threshold")
            interval = body.get("interval")
            job_id = _start_job(
                lambda progress, set_partial: score.score_video(
                    source, threshold or score.FAMILIARITY_THRESHOLD, progress, set_partial, interval=interval
                )
            )
            self._json(200, {"job_id": job_id})
        elif path == "/api/score/confirm":
            self._confirm_detection(body)
        elif path == "/api/faces/capture":
            self._capture_face(body)
        else:
            self._respond(404, "text/plain", b"Not found")

    def _confirm_detection(self, body: dict):
        detection = body["detection"]
        conn = db.connect()
        person_uuid = body.get("person_uuid")
        new_name = body.get("new_person_name")
        if new_name:
            person_uuid = db.create_person(conn, new_name)
        db.insert_face(
            conn,
            source=body.get("video", "review"),
            seconds=detection["seconds"],
            confidence=detection["confidence"],
            thumbnail_path=detection["thumbnail"],
            embedding_blob=embedding_to_bytes(np.array(detection["embedding"], dtype=np.float32)),
            person_uuid=person_uuid,
        )
        conn.close()
        self._json(200, {"person_uuid": person_uuid})

    def _capture_face(self, body: dict):
        """Same idea as _confirm_detection, but for a face captured live by
        /watch — there's no pre-existing report/thumbnail file on disk (review.html's
        flow already wrote one during scoring; this one is a fresh crop from the
        browser's own <video> frame), so the thumbnail JPEG arrives as base64 and
        gets written to thumbnail_dir() here instead of just referenced by name."""
        conn = db.connect()
        person_uuid = body.get("person_uuid")
        new_name = body.get("new_person_name")
        if new_name:
            person_uuid = db.create_person(conn, new_name)
        thumb_name = f"watch_{db.new_uuid()[:8]}.jpg"
        (thumbnail_dir() / thumb_name).write_bytes(base64.b64decode(body["thumbnail"]))
        face_uuid = db.insert_face(
            conn,
            source="watch",
            seconds=0.0,
            confidence=body.get("confidence", 0.0),
            thumbnail_path=thumb_name,
            embedding_blob=embedding_to_bytes(np.array(body["embedding"], dtype=np.float32)),
            person_uuid=person_uuid,
        )
        conn.close()
        self._json(200, {"uuid": face_uuid, "person_uuid": person_uuid})


def _free_port(port: int) -> None:
    result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
    pids = [p for p in result.stdout.strip().split() if p]
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    if pids:
        for _ in range(20):
            time.sleep(0.1)
            check = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            if not check.stdout.strip():
                break


if __name__ == "__main__":
    _free_port(PORT)
    print(f"Open http://localhost:{PORT}  (label) and http://localhost:{PORT}/review (score a video)")

    class _Server(ThreadingHTTPServer):
        allow_reuse_address = True

    server = _Server(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

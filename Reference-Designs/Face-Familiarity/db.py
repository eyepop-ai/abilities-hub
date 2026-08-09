#!/usr/bin/env python3.12
"""SQLite-backed face/person library for Face-Familiarity (see CLAUDE.md)."""
import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path("library.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    uuid TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS faces (
    uuid TEXT PRIMARY KEY,
    person_uuid TEXT REFERENCES people(uuid),
    source TEXT NOT NULL,
    seconds REAL NOT NULL,
    confidence REAL NOT NULL,
    thumbnail_path TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def new_uuid() -> str:
    return uuid.uuid4().hex


def create_person(conn: sqlite3.Connection, name: str) -> str:
    person_uuid = new_uuid()
    conn.execute(
        "INSERT INTO people (uuid, name, created_at) VALUES (?, ?, ?)",
        (person_uuid, name, time.time()),
    )
    conn.commit()
    return person_uuid


def insert_face(
    conn: sqlite3.Connection, *, source: str, seconds: float, confidence: float,
    thumbnail_path: str, embedding_blob: bytes, person_uuid: str | None = None,
) -> str:
    face_uuid = new_uuid()
    conn.execute(
        "INSERT INTO faces "
        "(uuid, person_uuid, source, seconds, confidence, thumbnail_path, embedding, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (face_uuid, person_uuid, source, seconds, confidence, thumbnail_path, embedding_blob, time.time()),
    )
    conn.commit()
    return face_uuid


def rename_person(conn: sqlite3.Connection, person_uuid: str, name: str) -> None:
    conn.execute("UPDATE people SET name = ? WHERE uuid = ?", (name, person_uuid))
    conn.commit()


def delete_person(conn: sqlite3.Connection, person_uuid: str) -> None:
    """Deletes the person; their faces go back to unassigned rather than being deleted."""
    conn.execute("UPDATE faces SET person_uuid = NULL WHERE person_uuid = ?", (person_uuid,))
    conn.execute("DELETE FROM people WHERE uuid = ?", (person_uuid,))
    conn.commit()


def assign_faces(conn: sqlite3.Connection, face_uuids: list[str], person_uuid: str | None) -> None:
    conn.executemany(
        "UPDATE faces SET person_uuid = ? WHERE uuid = ?",
        [(person_uuid, face_uuid) for face_uuid in face_uuids],
    )
    conn.commit()


def assign_face_pairs(conn: sqlite3.Connection, pairs: list[tuple[str, str]]) -> None:
    """Batch-assign faces to potentially different people: (person_uuid, face_uuid) pairs."""
    conn.executemany("UPDATE faces SET person_uuid = ? WHERE uuid = ?", pairs)
    conn.commit()


def list_people(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT p.uuid, p.name, COUNT(f.uuid) AS face_count "
        "FROM people p LEFT JOIN faces f ON f.person_uuid = p.uuid "
        "GROUP BY p.uuid ORDER BY p.name COLLATE NOCASE"
    ).fetchall()


def list_faces(conn: sqlite3.Connection, person_uuid: str | None) -> list[sqlite3.Row]:
    """`person_uuid=None` returns every face; pass "unassigned" for unlabeled faces."""
    if person_uuid == "unassigned":
        return conn.execute(
            "SELECT * FROM faces WHERE person_uuid IS NULL ORDER BY created_at DESC"
        ).fetchall()
    if person_uuid is None:
        return conn.execute("SELECT * FROM faces ORDER BY created_at DESC").fetchall()
    return conn.execute(
        "SELECT * FROM faces WHERE person_uuid = ? ORDER BY created_at DESC", (person_uuid,)
    ).fetchall()


def get_face(conn: sqlite3.Connection, face_uuid: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM faces WHERE uuid = ?", (face_uuid,)).fetchone()


def labeled_faces(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every face that's been assigned to a person, with that person's name — used to build
    per-person centroid embeddings for matching in score.py."""
    return conn.execute(
        "SELECT f.person_uuid, f.embedding, p.name FROM faces f "
        "JOIN people p ON p.uuid = f.person_uuid WHERE f.person_uuid IS NOT NULL"
    ).fetchall()

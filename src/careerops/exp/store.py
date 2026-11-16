"""Experience bank storage. Tables are prefixed exp_."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime

import numpy as np

from careerops.domain import ExperienceItem, ResumeVersion

SCHEMA = """
CREATE TABLE IF NOT EXISTS exp_items (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    approved INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS exp_embeddings (
    item_id TEXT PRIMARY KEY REFERENCES exp_items(id) ON DELETE CASCADE,
    text_hash TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS exp_resumes (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def embedding_text(item: ExperienceItem) -> str:
    parts = [item.title, item.text, *item.metrics, *item.skills]
    return " ".join(p for p in parts if p)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def save_item(conn: sqlite3.Connection, item: ExperienceItem, *, approved: bool = False) -> None:
    """Insert or replace an item. Editing an item resets approval unless approved=True is passed."""
    init(conn)
    now = _now()
    conn.execute(
        "INSERT INTO exp_items (id, data, approved, created_at, updated_at) VALUES (?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET data=excluded.data, approved=excluded.approved, "
        "updated_at=excluded.updated_at",
        (item.id, item.model_dump_json(), int(approved), now, now),
    )
    conn.commit()


def get_item(conn: sqlite3.Connection, item_id: str) -> ExperienceItem | None:
    init(conn)
    row = conn.execute("SELECT data FROM exp_items WHERE id=?", (item_id,)).fetchone()
    return ExperienceItem.model_validate_json(row["data"]) if row else None


def list_items(
    conn: sqlite3.Connection, *, approved: bool | None = None
) -> list[ExperienceItem]:
    """List items. approved=True or False filters by review status. None returns all."""
    init(conn)
    sql = "SELECT data FROM exp_items"
    args: tuple = ()
    if approved is not None:
        sql += " WHERE approved=?"
        args = (int(approved),)
    rows = conn.execute(sql + " ORDER BY created_at, id", args).fetchall()
    return [ExperienceItem.model_validate_json(r["data"]) for r in rows]


def is_approved(conn: sqlite3.Connection, item_id: str) -> bool:
    init(conn)
    row = conn.execute("SELECT approved FROM exp_items WHERE id=?", (item_id,)).fetchone()
    return bool(row and row["approved"])


def set_approved(conn: sqlite3.Connection, item_id: str, approved: bool = True) -> None:
    init(conn)
    cur = conn.execute(
        "UPDATE exp_items SET approved=?, updated_at=? WHERE id=?",
        (int(approved), _now(), item_id),
    )
    if cur.rowcount == 0:
        raise KeyError(item_id)
    conn.commit()


def approved_ids(conn: sqlite3.Connection) -> set[str]:
    init(conn)
    rows = conn.execute("SELECT id FROM exp_items WHERE approved=1").fetchall()
    return {r["id"] for r in rows}


def delete_item(conn: sqlite3.Connection, item_id: str) -> None:
    init(conn)
    conn.execute("DELETE FROM exp_items WHERE id=?", (item_id,))
    conn.commit()


def put_embedding(conn: sqlite3.Connection, item: ExperienceItem, vec: np.ndarray) -> None:
    init(conn)
    arr = np.asarray(vec, dtype=np.float32)
    conn.execute(
        "INSERT OR REPLACE INTO exp_embeddings (item_id, text_hash, dim, vec) VALUES (?,?,?,?)",
        (item.id, _hash(embedding_text(item)), arr.shape[0], arr.tobytes()),
    )
    conn.commit()


def get_embedding(conn: sqlite3.Connection, item: ExperienceItem) -> np.ndarray | None:
    """Return the stored vector, or None if missing or stale (item text changed)."""
    init(conn)
    row = conn.execute("SELECT * FROM exp_embeddings WHERE item_id=?", (item.id,)).fetchone()
    if row is None or row["text_hash"] != _hash(embedding_text(item)):
        return None
    return np.frombuffer(row["vec"], dtype=np.float32).copy()


def save_resume(conn: sqlite3.Connection, resume: ResumeVersion) -> None:
    init(conn)
    conn.execute(
        "INSERT OR REPLACE INTO exp_resumes (id, job_id, data, created_at) VALUES (?,?,?,?)",
        (resume.id, resume.job_id, resume.model_dump_json(), resume.created_at.isoformat()),
    )
    conn.commit()


def get_resume(conn: sqlite3.Connection, resume_id: str) -> ResumeVersion | None:
    init(conn)
    row = conn.execute("SELECT data FROM exp_resumes WHERE id=?", (resume_id,)).fetchone()
    return ResumeVersion.model_validate_json(row["data"]) if row else None

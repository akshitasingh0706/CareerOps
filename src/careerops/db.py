"""SQLite connection helper. Each module creates and owns its own tables.

Table names are prefixed by module: ingest_*, exp_*, crm_*, rank_*.
Store pydantic objects as JSON in a `data` column plus the few columns you query on.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_PATH = "data/careerops.db"


def connect(path: str | None = None) -> sqlite3.Connection:
    p = path or os.environ.get("CAREEROPS_DB", DEFAULT_PATH)
    if p != ":memory:":
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn

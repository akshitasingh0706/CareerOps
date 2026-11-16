"""Storage and watchlist. Tables are prefixed ingest_."""
from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

import httpx

from careerops.domain import Company, Job, Seniority, Source
from careerops.ingest.connectors import RawJob, fetch_company_jobs
from careerops.ingest.extract import extract_fields
from careerops.llm import LLM

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_companies (
    id TEXT PRIMARY KEY, watched INTEGER NOT NULL DEFAULT 0, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, source_id TEXT NOT NULL,
    company_id TEXT NOT NULL, canonical_id TEXT, closed INTEGER NOT NULL DEFAULT 0,
    first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, data TEXT NOT NULL, raw TEXT
);
CREATE INDEX IF NOT EXISTS ingest_jobs_company ON ingest_jobs(company_id);
CREATE TABLE IF NOT EXISTS ingest_pair_judgments (
    job_a TEXT NOT NULL, job_b TEXT NOT NULL, same INTEGER NOT NULL, reason TEXT NOT NULL,
    PRIMARY KEY (job_a, job_b)
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Create the ingest tables if they do not exist."""
    conn.executescript(SCHEMA)
    conn.commit()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_job(row: sqlite3.Row) -> Job:
    job = Job.model_validate_json(row["data"])
    job.canonical_id = row["canonical_id"]
    job.closed = bool(row["closed"])
    return job


def get_job(conn: sqlite3.Connection, job_id: str) -> Job | None:
    """Return one job or None."""
    row = conn.execute("SELECT * FROM ingest_jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def get_raw(conn: sqlite3.Connection, job_id: str) -> dict | None:
    """Return the raw source payload kept for re-extraction."""
    row = conn.execute("SELECT raw FROM ingest_jobs WHERE id = ?", (job_id,)).fetchone()
    return json.loads(row["raw"]) if row and row["raw"] else None


def upsert_job(conn: sqlite3.Connection, job: Job, raw: dict | None = None) -> bool:
    """Insert or update a job. Returns True if it is new.

    On update, fields the source lacks keep their stored (extracted) values,
    canonical_id is kept, and a closed job is reopened.
    """
    old = get_job(conn, job.id)
    now = _now()
    if old is None:
        conn.execute(
            "INSERT INTO ingest_jobs (id, source, source_id, company_id, canonical_id, closed,"
            " first_seen, last_seen, data, raw) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (job.id, job.source.value, job.source_id, job.company_id, job.canonical_id,
             int(job.closed), now, now, job.model_dump_json(),
             json.dumps(raw) if raw is not None else None),
        )
        conn.commit()
        return True
    merged = job.model_copy()
    if merged.seniority == Seniority.UNKNOWN:
        merged.seniority = old.seniority
    if merged.remote is None:
        merged.remote = old.remote
    if merged.salary_min is None and merged.salary_max is None:
        merged.salary_min, merged.salary_max = old.salary_min, old.salary_max
        merged.salary_currency = old.salary_currency
    merged.canonical_id = old.canonical_id
    merged.closed = False
    conn.execute(
        "UPDATE ingest_jobs SET data = ?, closed = 0, last_seen = ?, raw = COALESCE(?, raw)"
        " WHERE id = ?",
        (merged.model_dump_json(), now, json.dumps(raw) if raw is not None else None, job.id),
    )
    conn.commit()
    return False


def save_job(conn: sqlite3.Connection, job: Job) -> None:
    """Write back a changed job (for example after extraction) without touching raw."""
    conn.execute(
        "UPDATE ingest_jobs SET data = ?, canonical_id = ?, closed = ? WHERE id = ?",
        (job.model_dump_json(), job.canonical_id, int(job.closed), job.id),
    )
    conn.commit()


def list_jobs(conn: sqlite3.Connection, *, company_id: str | None = None,
              source: Source | None = None, include_closed: bool = False,
              canonical_only: bool = False) -> list[Job]:
    """List stored jobs, newest first seen first."""
    sql, args = "SELECT * FROM ingest_jobs WHERE 1=1", []
    if company_id:
        sql += " AND company_id = ?"
        args.append(company_id)
    if source:
        sql += " AND source = ?"
        args.append(source.value)
    if not include_closed:
        sql += " AND closed = 0"
    if canonical_only:
        sql += " AND (canonical_id IS NULL OR canonical_id = id)"
    sql += " ORDER BY first_seen DESC, id"
    return [_row_to_job(r) for r in conn.execute(sql, args)]


def mark_closed(conn: sqlite3.Connection, job_ids: Iterable[str]) -> int:
    """Mark jobs closed. Returns how many changed."""
    n = 0
    for jid in job_ids:
        n += conn.execute("UPDATE ingest_jobs SET closed = 1 WHERE id = ? AND closed = 0",
                          (jid,)).rowcount
    conn.commit()
    return n


def close_missing(conn: sqlite3.Connection, company_id: str, source: Source,
                  seen_ids: set[str]) -> list[str]:
    """Close open jobs of this company and source that are not in seen_ids."""
    rows = conn.execute(
        "SELECT id FROM ingest_jobs WHERE company_id = ? AND source = ? AND closed = 0",
        (company_id, source.value)).fetchall()
    gone = [r["id"] for r in rows if r["id"] not in seen_ids]
    mark_closed(conn, gone)
    return gone


def add_company(conn: sqlite3.Connection, company: Company) -> None:
    """Add or replace a company. Set watched=True to poll it."""
    conn.execute(
        "INSERT INTO ingest_companies (id, watched, data) VALUES (?,?,?)"
        " ON CONFLICT(id) DO UPDATE SET watched = excluded.watched, data = excluded.data",
        (company.id, int(company.watched), company.model_dump_json()))
    conn.commit()


def list_companies(conn: sqlite3.Connection, *, watched_only: bool = False) -> list[Company]:
    """List companies."""
    sql = "SELECT data FROM ingest_companies"
    if watched_only:
        sql += " WHERE watched = 1"
    return [Company.model_validate_json(r["data"]) for r in conn.execute(sql + " ORDER BY id")]


def add_to_watchlist(conn: sqlite3.Connection, company: Company) -> None:
    """Store a company and watch it."""
    add_company(conn, company.model_copy(update={"watched": True}))


def ingest_jobs(conn: sqlite3.Connection, pairs: list[RawJob], llm: LLM | None = None) -> list[Job]:
    """Store (job, raw) pairs. Returns the jobs that were new.

    If an llm is given, new jobs get seniority, remote and salary filled when missing.
    """
    new: list[Job] = []
    for job, raw in pairs:
        if get_job(conn, job.id) is None:
            if llm is not None:
                job = extract_fields(job, llm)
            new.append(job)
        upsert_job(conn, job, raw)
    return new


def poll_watchlist(
    conn: sqlite3.Connection,
    *,
    client: httpx.Client | None = None,
    llm: LLM | None = None,
    fetcher: Callable[[Company, httpx.Client | None], list[RawJob]] = fetch_company_jobs,
) -> list[Job]:
    """Poll every watched company. Store jobs, close vanished ones, return new jobs.

    A company whose fetch fails is skipped and its jobs are left open.
    """
    new: list[Job] = []
    for company in list_companies(conn, watched_only=True):
        try:
            pairs = fetcher(company, client)
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("poll failed for %s: %s", company.id, exc)
            continue
        new.extend(ingest_jobs(conn, pairs, llm))
        if company.ats:
            close_missing(conn, company.id, company.ats, {j.id for j, _ in pairs})
    return new

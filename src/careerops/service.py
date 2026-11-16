"""Glue between the modules. The CLI and the API both call these functions.

Each call opens its own SQLite connection, so it is safe to use from web worker threads.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from careerops import ingest
from careerops.crm.service import CRM
from careerops.db import connect
from careerops.domain import (
    Company,
    Feedback,
    FeedbackAction,
    Job,
    Source,
    Stage,
    UserProfile,
)
from careerops.exp import store as exp_store
from careerops.llm import Embedder, FakeLLM, HashEmbedder
from careerops.rank import FeedbackStore, RankConfig, rank_jobs
from careerops.rank.ranker import RankResult

load_dotenv()


def profile_path() -> Path:
    return Path(os.environ.get("CAREEROPS_PROFILE", "data/profile.json"))


def load_profile() -> UserProfile:
    p = profile_path()
    if p.exists():
        return UserProfile.model_validate_json(p.read_text())
    return UserProfile()


def save_profile(profile: UserProfile) -> UserProfile:
    p = profile_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(profile.model_dump_json(indent=2))
    return profile


def get_embedder() -> Embedder:
    """Offline hashing embedder by default. Set CAREEROPS_EMBEDDER=fastembed for real embeddings.

    The fastembed model is downloaded from Hugging Face on first use.
    """
    if os.environ.get("CAREEROPS_EMBEDDER") == "fastembed":
        from careerops.embed import FastEmbedder

        return FastEmbedder()
    return HashEmbedder()


@contextmanager
def session() -> Iterator:
    """Open the database, make sure every module's tables exist, and close it afterwards."""
    conn = connect()
    ingest.init_db(conn)
    exp_store.init(conn)
    crm = CRM(conn=conn)
    try:
        yield conn, crm
    finally:
        conn.close()


# ---- companies and jobs ----------------------------------------------------------------


def watch_company(ats: str, token: str, name: str | None = None) -> Company:
    source = Source(ats)
    if source not in (Source.GREENHOUSE, Source.LEVER, Source.ASHBY):
        raise ValueError("ats must be greenhouse, lever or ashby")
    company = Company(
        id=f"{ats}-{token}".lower(), name=name or token, ats=source, board_token=token, watched=True
    )
    with session() as (conn, _):
        ingest.add_to_watchlist(conn, company)
    return company


def poll() -> list[Job]:
    with session() as (conn, _):
        return ingest.poll_watchlist(conn)


def companies() -> list[Company]:
    with session() as (conn, _):
        return ingest.list_companies(conn)


def jobs() -> list[Job]:
    with session() as (conn, _):
        return ingest.list_jobs(conn, canonical_only=True)


# ---- ranking ---------------------------------------------------------------------------


def _skipped_ids(store: FeedbackStore) -> set[str]:
    return {r.feedback.job_id for r in store.rows() if r.feedback.action == FeedbackAction.SKIPPED}


def rank(limit: int | None = None) -> RankResult:
    """Rank open jobs for the saved profile. No LLM calls: explanations are off here."""
    with session() as (conn, _):
        store = FeedbackStore(conn)
        skipped = _skipped_ids(store)
        candidates = [j for j in ingest.list_jobs(conn, canonical_only=True) if j.id not in skipped]
        items = exp_store.list_items(conn, approved=True)
        result = rank_jobs(
            candidates,
            load_profile(),
            items,
            FakeLLM(),
            get_embedder(),
            feedback=store,
            config=RankConfig(explain_top=0),
        )
    if limit is not None:
        result.ranked = result.ranked[:limit]
    return result


def record_feedback(job_id: str, action: FeedbackAction, reason: str | None = None) -> None:
    """Log feedback with the job's current component scores."""
    with session() as (conn, _):
        job = ingest.get_job(conn, job_id)
        if job is None:
            raise KeyError(job_id)
        items = exp_store.list_items(conn, approved=True)
        store = FeedbackStore(conn)
        scored = rank_jobs(
            [job],
            load_profile(),
            items,
            FakeLLM(),
            get_embedder(),
            config=RankConfig(explain_top=0),
        )
        entry = (scored.ranked + scored.filtered_out)[0]
        store.record(
            Feedback(job_id=job_id, action=action, at=datetime.now(), reason=reason), entry.fit
        )


def save_job(job_id: str):
    """Save a job as an application in stage SAVED and log positive feedback."""
    with session() as (conn, crm):
        if ingest.get_job(conn, job_id) is None:
            raise KeyError(job_id)
        existing = crm.application_for_job(job_id)
        if existing:
            return existing
        app = crm.create_application(job_id)
    record_feedback(job_id, FeedbackAction.SAVED)
    return app


def skip_job(job_id: str, reason: str | None = None) -> None:
    record_feedback(job_id, FeedbackAction.SKIPPED, reason)


# ---- applications ----------------------------------------------------------------------


def applications() -> list[dict]:
    with session() as (conn, crm):
        out = []
        for app in crm.list_applications():
            job = ingest.get_job(conn, app.job_id)
            out.append(
                {
                    "application": app.model_dump(mode="json"),
                    "job": job.model_dump(mode="json") if job else None,
                }
            )
        return out


def advance(app_id: str, stage: Stage, note: str = ""):
    with session() as (_, crm):
        return crm.advance(app_id, stage, note=note)


def pipeline() -> dict[str, int]:
    with session() as (_, crm):
        return {s.value: n for s, n in crm.pipeline_summary().items()}


def due_followups(today=None) -> list[dict]:
    from dataclasses import asdict
    from datetime import date

    with session() as (_, crm):
        return [json.loads(json.dumps(asdict(d), default=str)) for d in crm.due(today or date.today())]

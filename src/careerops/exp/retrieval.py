"""Pick the experience items most relevant to a job. Scoring is done in code."""
from __future__ import annotations

import re
import sqlite3

import numpy as np
from pydantic import BaseModel

from careerops.domain import ExperienceItem
from careerops.exp import store
from careerops.llm import Embedder

EMBED_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4


class ScoredItem(BaseModel):
    item: ExperienceItem
    score: float
    embedding_score: float
    keyword_score: float
    matched_skills: list[str]


def matched_skills(job_description: str, skills: list[str]) -> list[str]:
    """Skills that appear in the job text as whole terms, case-insensitive."""
    found = []
    for s in skills:
        s = s.strip()
        if s and re.search(rf"(?<![\w+#]){re.escape(s)}(?![\w+#])", job_description, re.IGNORECASE):
            found.append(s)
    return found


def _vectors(
    conn: sqlite3.Connection, items: list[ExperienceItem], embedder: Embedder
) -> np.ndarray:
    vecs: list[np.ndarray | None] = []
    for it in items:
        v = store.get_embedding(conn, it)
        vecs.append(v if v is not None and v.shape[0] == embedder.dim else None)
    missing = [i for i, v in enumerate(vecs) if v is None]
    if missing:
        new = embedder.embed([store.embedding_text(items[i]) for i in missing])
        for i, v in zip(missing, new, strict=True):
            store.put_embedding(conn, items[i], v)
            vecs[i] = v
    return np.vstack(vecs)


def select_items(
    job_description: str,
    k: int,
    embedder: Embedder,
    *,
    conn: sqlite3.Connection,
    approved_only: bool = True,
) -> list[ScoredItem]:
    """Return up to k items with scores, best first.

    score = 0.6 * cosine similarity (clipped to 0..1) + 0.4 * share of the item's skills
    found in the job text. Only approved items are used unless approved_only=False.
    """
    items = store.list_items(conn, approved=True if approved_only else None)
    if not items or k <= 0:
        return []
    q = embedder.embed([job_description])[0]
    mat = _vectors(conn, items, embedder)
    sims = np.clip(mat @ q, 0.0, 1.0)
    scored = []
    for item, sim in zip(items, sims, strict=True):
        hits = matched_skills(job_description, item.skills)
        kw = len(hits) / len(item.skills) if item.skills else 0.0
        score = EMBED_WEIGHT * float(sim) + KEYWORD_WEIGHT * kw
        scored.append(ScoredItem(item=item, score=score, embedding_score=float(sim),
                                 keyword_score=kw, matched_skills=hits))
    scored.sort(key=lambda s: (-s.score, s.item.id))
    return scored[:k]

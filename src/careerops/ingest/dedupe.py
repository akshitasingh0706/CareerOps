"""Dedupe. Code scores pairs and makes the merge. The LLM only judges borderline pairs.

Steps: normalize, block on company, score each pair (embedding cosine plus title similarity),
merge above HIGH, split below LOW, ask the judge in between. Then set canonical_id.
"""
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations

import numpy as np
from pydantic import BaseModel

from careerops.domain import Job, Source
from careerops.ingest.store import init_db, list_jobs, save_job
from careerops.llm import CHEAP_MODEL, LLM, Embedder

HIGH = 0.80
LOW = 0.50
COSINE_WEIGHT = 0.3
TITLE_WEIGHT = 0.7
LOCATION_PENALTY = 0.25

_COMPANY_SUFFIXES = {"inc", "llc", "ltd", "corp", "corporation", "co", "company", "gmbh", "plc",
                     "limited", "incorporated", "the"}
_TITLE_ABBR = {
    "sr": "senior", "jr": "junior", "ml": "machine learning", "swe": "software engineer",
    "sde": "software engineer", "eng": "engineer", "engr": "engineer", "mgr": "manager",
    "dev": "developer", "devops": "devops", "ai": "ai", "fe": "frontend", "be": "backend",
    "fullstack": "full stack", "ds": "data scientist", "pm": "product manager",
    "vp": "vice president", "assoc": "associate", "i": "1", "ii": "2", "iii": "3",
}
_TITLE_NOISE = {"remote", "hybrid", "onsite", "on-site", "us", "usa", "uk", "emea", "m", "f", "d",
                "w", "x", "all", "genders", "and", "of", "the", "a", "for", "at"}
_LEVELS = {"intern", "junior", "senior", "staff", "principal", "lead", "manager", "director",
           "head", "vice", "1", "2", "3", "associate"}
_CITY_ALIAS = {"sf": "san francisco", "san fran": "san francisco", "nyc": "new york",
               "new york city": "new york", "ny": "new york", "la": "los angeles",
               "bay area": "san francisco", "sea": "seattle", "atx": "austin"}
_STATE = {"ca", "ny", "wa", "tx", "ma", "il", "co", "uk", "us", "usa", "united", "states",
          "kingdom", "area", "metro", "greater", "hybrid", "on", "site", "onsite", "or", "and"}


def normalize_company(name: str) -> str:
    """Lowercase, strip punctuation and legal suffixes."""
    toks = re.sub(r"[^a-z0-9 ]+", " ", name.lower().replace("&", " and ")).split()
    toks = [t for t in toks if t not in _COMPANY_SUFFIXES]
    return " ".join(toks)


def normalize_title(title: str) -> str:
    """Lowercase, expand abbreviations, drop parentheses and noise words."""
    t = re.sub(r"\([^)]*\)", " ", title.lower())
    t = t.replace("&", " and ").replace("/", " ").replace(",", " ")
    t = re.sub(r"[^a-z0-9+# ]+", " ", t)
    out: list[str] = []
    for tok in t.split():
        if tok in _TITLE_NOISE:
            continue
        out.extend(_TITLE_ABBR.get(tok, tok).split())
    return " ".join(out)


def _level_tokens(norm_title: str) -> set[str]:
    return {t for t in norm_title.split() if t in _LEVELS}


def _location_tokens(loc: str | None) -> set[str]:
    if not loc:
        return set()
    s = re.sub(r"[^a-z ]+", " ", loc.lower())
    s = f" {s} "
    for alias, full in sorted(_CITY_ALIAS.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(f" {alias} ", f" {full} ")
    return {t for t in s.split() if t not in _STATE and len(t) > 1}


def location_compatible(a: Job, b: Job) -> bool:
    """False only when both jobs name places and the places share nothing."""
    ta, tb = _location_tokens(a.location), _location_tokens(b.location)
    if not ta or not tb:
        return True
    if "remote" in ta and "remote" in tb:
        return True
    return bool(ta & tb)


def title_similarity(a: str, b: str) -> float:
    """0..1 similarity of normalized titles: mean of token Jaccard and sequence ratio."""
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    sa, sb = set(na.split()), set(nb.split())
    jac = len(sa & sb) / len(sa | sb)
    return (jac + SequenceMatcher(None, na, nb).ratio()) / 2


def _embed_text(job: Job) -> str:
    return f"{job.title} {job.title} {job.description[:1200]}"


@dataclass
class PairScore:
    cosine: float
    title_sim: float
    score: float
    location_ok: bool
    level_conflict: bool


def score_pair(a: Job, b: Job, va: np.ndarray, vb: np.ndarray) -> PairScore:
    """Score a pair. Embeddings are L2-normalized, so cosine is a dot product."""
    cos = float(np.clip(np.dot(va, vb), 0.0, 1.0))
    ts = title_similarity(a.title, b.title)
    la, lb = _level_tokens(normalize_title(a.title)), _level_tokens(normalize_title(b.title))
    conflict = bool(la and lb and la != lb)
    loc_ok = location_compatible(a, b)
    score = COSINE_WEIGHT * cos + TITLE_WEIGHT * ts
    if not loc_ok:
        score -= LOCATION_PENALTY
    if bool(la) != bool(lb):
        score -= 0.15  # one title states a level and the other does not
    return PairScore(cos, ts, max(score, 0.0), loc_ok, conflict)


class SamePosting(BaseModel):
    """The judge's answer for a borderline pair."""
    same: bool
    reason: str


JUDGE_SYSTEM = (
    "You decide whether two job listings describe the same open position at the same company. "
    "Different locations, levels or teams mean different positions. "
    "Cross-posted copies of one role are the same."
)


def _fmt(j: Job) -> str:
    return (f"Title: {j.title}\nCompany: {j.company_name}\nLocation: {j.location or 'n/a'}\n"
            f"Source: {j.source.value}\nDescription: {j.description[:600]}")


def judge_pair(a: Job, b: Job, llm: LLM) -> SamePosting:
    """Ask the LLM whether a borderline pair is the same posting."""
    prompt = f"Listing A:\n{_fmt(a)}\n\nListing B:\n{_fmt(b)}"
    return llm.structured(SamePosting, JUDGE_SYSTEM, prompt, model=CHEAP_MODEL)


def decide(ps: PairScore, high: float = HIGH, low: float = LOW) -> str:
    """Return 'merge', 'split' or 'judge'. Thresholds live here, not in the LLM."""
    if ps.level_conflict:
        return "split"
    if ps.score >= high and ps.location_ok:
        return "merge"
    if ps.score < low:
        return "split"
    return "judge"


def are_duplicates(a: Job, b: Job, embedder: Embedder, llm: LLM | None = None,
                   high: float = HIGH, low: float = LOW) -> bool:
    """Full decision for one pair. Without an llm, borderline pairs count as different."""
    if normalize_company(a.company_name) != normalize_company(b.company_name):
        return False
    va, vb = embedder.embed([_embed_text(a), _embed_text(b)])
    verdict = decide(score_pair(a, b, va, vb), high, low)
    if verdict == "judge":
        return judge_pair(a, b, llm).same if llm else False
    return verdict == "merge"


_SOURCE_RANK = {Source.GREENHOUSE: 0, Source.LEVER: 0, Source.ASHBY: 0, Source.CAREER_PAGE: 1,
                Source.AGGREGATOR: 2, Source.MANUAL: 3, Source.BROWSER_CAPTURE: 3,
                Source.EMAIL_ALERT: 4}


def _representative(jobs: list[Job]) -> Job:
    """Prefer ATS sources, then longer descriptions, then lowest id."""
    return min(jobs, key=lambda j: (_SOURCE_RANK[j.source], -len(j.description), j.id))


def dedupe_jobs(
    jobs: list[Job],
    embedder: Embedder,
    llm: LLM | None = None,
    *,
    high: float = HIGH,
    low: float = LOW,
    cached: dict[tuple[str, str], bool] | None = None,
    on_judgment: callable | None = None,
) -> dict[str, str]:
    """Return {job_id: canonical_id} for all jobs. Singletons map to themselves."""
    cached = cached or {}
    parent = {j.id: j.id for j in jobs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    blocks: dict[str, list[Job]] = defaultdict(list)
    for j in jobs:
        blocks[normalize_company(j.company_name)].append(j)
    for block in blocks.values():
        if len(block) < 2:
            continue
        vecs = embedder.embed([_embed_text(j) for j in block])
        for i, k in combinations(range(len(block)), 2):
            a, b = block[i], block[k]
            verdict = decide(score_pair(a, b, vecs[i], vecs[k]), high, low)
            if verdict == "judge":
                key = tuple(sorted((a.id, b.id)))
                if key in cached:
                    verdict = "merge" if cached[key] else "split"
                elif llm is not None:
                    res = judge_pair(a, b, llm)
                    if on_judgment:
                        on_judgment(key, res)
                    verdict = "merge" if res.same else "split"
                else:
                    verdict = "split"
            if verdict == "merge":
                parent[find(a.id)] = find(b.id)
    clusters: dict[str, list[Job]] = defaultdict(list)
    for j in jobs:
        clusters[find(j.id)].append(j)
    out: dict[str, str] = {}
    for members in clusters.values():
        rep = _representative(members)
        for m in members:
            out[m.id] = rep.id
    return out


def dedupe_store(conn: sqlite3.Connection, embedder: Embedder, llm: LLM | None = None,
                 *, high: float = HIGH, low: float = LOW) -> dict[str, str]:
    """Recompute canonical_id for all open stored jobs. Judge answers are cached in the db."""
    init_db(conn)
    jobs = list_jobs(conn)
    cached = {(r["job_a"], r["job_b"]): bool(r["same"])
              for r in conn.execute("SELECT * FROM ingest_pair_judgments")}

    def record(key: tuple[str, str], res: SamePosting) -> None:
        conn.execute("INSERT OR REPLACE INTO ingest_pair_judgments VALUES (?,?,?,?)",
                     (key[0], key[1], int(res.same), res.reason))

    mapping = dedupe_jobs(jobs, embedder, llm, high=high, low=low, cached=cached,
                          on_judgment=record)
    for j in jobs:
        if j.canonical_id != mapping[j.id]:
            j.canonical_id = mapping[j.id]
            save_job(conn, j)
    conn.commit()
    return mapping

"""Ranking eval on synthetic jobs with a hidden preference rule.

The simulated user cares about topic, pay and remote work far more than the rules weights
assume. We feed feedback to the ranker and compare NDCG@10 of rules-only against personalized.
"""
from __future__ import annotations

import random
import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from careerops.domain import (
    ExperienceItem,
    ExperienceKind,
    Feedback,
    FeedbackAction,
    FitScore,
    Job,
    Seniority,
    Source,
    UserProfile,
)
from careerops.llm import HashEmbedder
from careerops.rank import (
    FeedbackStore,
    Personalizer,
    RankConfig,
    ndcg_at_k,
    overall_score,
    score_components,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
TOPICS = {
    "devtools": "developer tools compilers build systems sdk",
    "mlinfra": "machine learning infrastructure training pipelines gpu",
    "fintech": "payments ledger banking fraud",
    "adtech": "advertising bidding campaigns tracking",
    "health": "clinical patient records scheduling",
    "retail": "ecommerce catalog inventory checkout",
}
SKILLS = ["python", "sql", "go", "react", "kubernetes", "spark", "rust", "java"]
LOCS = ["New York, NY", "San Francisco, CA", "Austin, TX", "London, UK"]
N_JOBS = 160  # first 100 are shown to the user, last 60 are held out for scoring
N_TRAIN_POOL = 100
EVENT_COUNTS = (10, 30, 60)
SEEDS = range(8)

PROFILE = UserProfile(
    target_titles=["Senior Software Engineer"],
    min_seniority=Seniority.MID,
    max_seniority=Seniority.STAFF,
    locations=["New York"],
    remote_ok=True,
    min_salary=150_000,
    interests=["developer tools", "machine learning infrastructure"],
)
ITEMS = [
    ExperienceItem(
        id="e1", kind=ExperienceKind.ACCOMPLISHMENT, title="Platform engineer",
        text="Built python services and sql pipelines for analytics", skills=["python", "sql"],
    ),
]


def make_jobs(rng: random.Random) -> list[tuple[Job, dict]]:
    out = []
    for i in range(N_JOBS):
        topic = rng.choice(list(TOPICS))
        skills = rng.sample(SKILLS, 3)
        remote = rng.random() < 0.4
        sen = rng.choice([Seniority.JUNIOR, Seniority.MID, Seniority.SENIOR, Seniority.STAFF])
        salary = None if rng.random() < 0.15 else rng.randrange(100, 260, 10) * 1000
        loc = None if remote else rng.choice(LOCS)
        desc = f"We build {TOPICS[topic]}. You will use {' and '.join(skills)} every day."
        job = Job(
            id=f"j{i}", source=Source.MANUAL, source_id=str(i), url=f"http://x/{i}",
            company_id=f"c{i}", company_name=f"Co{i}", title=f"{sen.value} engineer {topic}",
            location=loc, remote=remote, description=desc, fetched_at=NOW,
            salary_min=salary, salary_max=salary, seniority=sen,
        )
        out.append((job, {"topic": topic, "remote": remote, "salary": salary or 0, "sen": sen}))
    return out


def hidden_utility(attrs: dict) -> float:
    """The user's real taste. It differs from the rules weights: it ignores skills,
    loves remote and pay, and wants a specific topic."""
    u = 0.0
    u += 2.0 if attrs["topic"] == "devtools" else 0.0
    u += 1.0 if attrs["topic"] == "mlinfra" else 0.0
    u += 2.0 if attrs["salary"] >= 190_000 else 0.0
    u += 1.5 if attrs["remote"] else 0.0
    u -= 1.0 if attrs["sen"] == Seniority.STAFF else 0.0
    return u


def _fit(job: Job, comps: dict[str, float]) -> FitScore:
    return FitScore(
        job_id=job.id, overall=overall_score(comps), components=comps, filters=[],
        explanation="", scorer_version="test",
    )


def run_seed(seed: int) -> dict[int, tuple[float, float]]:
    rng = random.Random(seed)
    jobs_attrs = make_jobs(rng)
    jobs = [j for j, _ in jobs_attrs]
    comps = score_components(jobs, PROFILE, ITEMS, HashEmbedder())
    util = [hidden_utility(a) for _, a in jobs_attrs]
    shift = min(util)
    test_idx = list(range(N_TRAIN_POOL, N_JOBS))
    rules = np.array([overall_score(c) for c in comps])
    cfg = RankConfig()

    def ndcg(scores: np.ndarray) -> float:
        order = sorted(test_idx, key=lambda i: -scores[i])
        return ndcg_at_k([util[i] - shift for i in order], 10)

    shown = rng.sample(range(N_TRAIN_POOL), N_TRAIN_POOL)
    store = FeedbackStore(sqlite3.connect(":memory:"))
    store.conn.row_factory = sqlite3.Row
    results: dict[int, tuple[float, float]] = {}
    base = ndcg(rules)
    for n, idx in enumerate(shown[: max(EVENT_COUNTS)], start=1):
        noisy = util[idx] + rng.gauss(0, 0.5)
        action = FeedbackAction.SAVED if noisy >= 2.0 else FeedbackAction.SKIPPED
        store.record(Feedback(job_id=jobs[idx].id, action=action, at=NOW), _fit(jobs[idx], comps[idx]))
        if n in EVENT_COUNTS:
            model = Personalizer.from_feedback(store.rows(), cfg.personalization)
            blended = np.array([model.blend(rules[i], comps[i])[0] for i in range(N_JOBS)])
            results[n] = (base, ndcg(blended))
    return results


@pytest.fixture(scope="module")
def eval_numbers() -> dict[int, tuple[float, float]]:
    runs = [run_seed(s) for s in SEEDS]
    return {
        n: (float(np.mean([r[n][0] for r in runs])), float(np.mean([r[n][1] for r in runs])))
        for n in EVENT_COUNTS
    }


def test_personalized_beats_rules_at_60_events(eval_numbers):
    print("\nNDCG@10 averaged over", len(SEEDS), "seeds,", N_JOBS - N_TRAIN_POOL, "held-out jobs")
    print("events  rules_only  personalized")
    for n in EVENT_COUNTS:
        rules, pers = eval_numbers[n]
        print(f"{n:>6}  {rules:10.3f}  {pers:12.3f}")
    rules60, pers60 = eval_numbers[60]
    assert pers60 > rules60


def test_personalization_does_not_hurt_much_early(eval_numbers):
    rules10, pers10 = eval_numbers[10]
    assert pers10 > rules10 - 0.10


def test_insufficient_data_is_reported():
    info = Personalizer.from_feedback([]).info
    assert info.status == "insufficient_data"
    assert "need at least" in info.needed

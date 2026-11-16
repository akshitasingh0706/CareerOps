import sqlite3
from datetime import UTC, datetime

import pytest

from careerops.domain import (
    ExperienceItem,
    ExperienceKind,
    Feedback,
    FeedbackAction,
    Job,
    Seniority,
    Source,
    UserProfile,
)
from careerops.llm import FakeLLM, HashEmbedder
from careerops.rank import (
    FeedbackStore,
    RankConfig,
    ScoringWeights,
    build_fit,
    explain_fit,
    overall_score,
    rank_jobs,
    score_components,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
ITEMS = [
    ExperienceItem(id="e1", kind=ExperienceKind.ACCOMPLISHMENT, title="Data platform",
                   text="Built python data pipelines", skills=["python", "sql"]),
    ExperienceItem(id="e2", kind=ExperienceKind.SKILL, title="Go", text="Wrote go services",
                   skills=["go"]),
]
PROFILE = UserProfile(
    target_titles=["Data Engineer"], min_seniority=Seniority.MID, max_seniority=Seniority.SENIOR,
    locations=["New York"], remote_ok=True, min_salary=140_000,
    interests=["data infrastructure"],
)


def job(i: str, **kw) -> Job:
    base = {
        "id": i, "source": Source.MANUAL, "source_id": i, "url": "u", "company_id": f"c{i}", "company_name": f"Co {i}",
        "title": "Data Engineer", "description": "python sql data infrastructure pipelines",
        "fetched_at": NOW, "seniority": Seniority.SENIOR, "location": "New York, NY", "remote": False,
        "salary_min": 150_000, "salary_max": 180_000,
    }
    base.update(kw)
    return Job(**base)


def fake(ids_by_req):
    return FakeLLM(structured={"ExplanationOut": {
        "explanation": "Strong match on python.",
        "requirements": [{"requirement": r, "item_ids": ids} for r, ids in ids_by_req.items()],
    }})


def test_components_are_in_range_and_ordered_sensibly():
    good = job("good")
    bad = job("bad", description="retail checkout cashier", location="Austin, TX",
              salary_max=80_000, salary_min=70_000, seniority=Seniority.PRINCIPAL, title="Cashier")
    cg, cb = score_components([good, bad], PROFILE, ITEMS, HashEmbedder())
    for c in (cg, cb):
        assert set(c) == {"skills", "seniority", "compensation", "location", "growth"}
        assert all(0 <= v <= 1 for v in c.values())
    for k in cg:
        assert cg[k] >= cb[k]
    assert cg["growth"] > cb["growth"]
    assert cb["location"] == 0.0 and cb["compensation"] == 0.0


def test_unknown_data_gives_neutral_components():
    c, = score_components([job("x", salary_min=None, salary_max=None, location=None, remote=None,
                               seniority=Seniority.UNKNOWN)], PROFILE, ITEMS, HashEmbedder())
    assert c["compensation"] == c["location"] == c["seniority"] == 0.5


def test_overall_is_weighted_sum():
    comps = {"skills": 1.0, "seniority": 0.0, "compensation": 0.0, "location": 0.0, "growth": 0.0}
    assert overall_score(comps, ScoringWeights(skills=1, seniority=1, compensation=1,
                                               location=1, growth=1)) == pytest.approx(0.2)
    assert overall_score(comps, ScoringWeights(skills=3, seniority=1, compensation=0,
                                               location=0, growth=0)) == pytest.approx(0.75)


def test_weights_validation():
    with pytest.raises(ValueError):
        ScoringWeights(skills=-1)


def test_invented_evidence_ids_are_dropped_and_flagged():
    llm = fake({"python": ["e1", "zzz"], "kubernetes": ["nope"], "go": ["e2", "e2"]})
    exp = explain_fit(llm, job("a"), PROFILE, ITEMS, {"skills": 0.5})
    assert exp.error is None
    assert exp.dropped_ids == ["zzz", "nope"]
    by_req = {m.requirement: m.item_ids for m in exp.matches}
    assert by_req == {"python": ["e1"], "kubernetes": [], "go": ["e2"]}
    lines = exp.evidence_lines()
    assert "kubernetes => (no supporting experience)" in lines
    assert "python => e1" in lines


def test_explain_falls_back_when_llm_fails():
    exp = explain_fit(FakeLLM(), job("a"), PROFILE, ITEMS, {"skills": 0.5, "growth": 0.9})
    assert exp.error and "growth" in exp.explanation


def test_llm_text_does_not_change_scores():
    comps = {"skills": 0.4, "seniority": 0.5, "compensation": 0.5, "location": 1.0, "growth": 0.2}
    a = build_fit(job("a"), comps, RankConfig(), [], None)
    exp = explain_fit(fake({"x": ["e1"]}), job("a"), PROFILE, ITEMS, comps)
    b = build_fit(job("a"), comps, RankConfig(), [], exp)
    assert a.overall == b.overall and b.evidence == ["x => e1"]


def test_rank_jobs_separates_filtered_and_sorts():
    jobs = [
        job("ok1"),
        job("ok2", description="retail checkout", title="Data Engineer"),
        job("avoid", company_name="Acme", company_id="acme"),
        job("lowpay", salary_min=50_000, salary_max=60_000),
        job("nosalary", salary_min=None, salary_max=None),
    ]
    prof = PROFILE.model_copy(update={"avoid_companies": ["acme"]})
    llm = fake({"python": ["e1"]})
    res = rank_jobs(jobs, prof, ITEMS, llm, HashEmbedder())
    assert {r.job.id for r in res.filtered_out} == {"avoid", "lowpay"}
    assert {r.job.id for r in res.ranked} == {"ok1", "ok2", "nosalary"}
    scores = [r.final_score for r in res.ranked]
    assert scores == sorted(scores, reverse=True)
    assert res.ranked[0].job.id == "ok1"
    nos = next(r for r in res.ranked if r.job.id == "nosalary")
    assert nos.unknown_filters == ["compensation"]
    avoid = next(r for r in res.filtered_out if r.job.id == "avoid")
    assert avoid.failed_filters == ["avoid_companies"] and avoid.explanation is None
    assert res.ranked[0].fit.evidence == ["python => e1"]
    assert res.personalization.status == "insufficient_data"
    assert res.ranked[0].learned_score is None


def test_explain_top_limits_llm_calls():
    jobs = [job(str(i)) for i in range(5)]
    llm = fake({"python": ["e1"]})
    rank_jobs(jobs, PROFILE, ITEMS, llm, HashEmbedder(), config=RankConfig(explain_top=2))
    assert len(llm.calls) == 2


def test_feedback_store_keeps_components_at_action_time():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store = FeedbackStore(conn)
    comps = {"skills": 0.1, "seniority": 0.2, "compensation": 0.3, "location": 0.4, "growth": 0.5}
    fit = build_fit(job("a"), comps, RankConfig(), [], None)
    store.record(Feedback(job_id="a", action=FeedbackAction.APPLIED, at=NOW, reason="nice"), fit)
    store.record(Feedback(job_id="b", action=FeedbackAction.SKIPPED, at=NOW), fit)
    rows = store.rows()
    assert store.count() == 2
    assert rows[0].components == comps and rows[0].label == 1 and rows[1].label == 0
    assert rows[0].feedback.reason == "nice"
    assert rows[0].rules_score == fit.overall and rows[0].scorer_version == fit.scorer_version
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert "rank_feedback" in tables


def test_personalization_activates_and_exposes_weights():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store = FeedbackStore(conn)
    # The user likes jobs with high growth and dislikes the rest.
    for i in range(20):
        g = 0.9 if i % 2 == 0 else 0.1
        comps = {"skills": 0.5, "seniority": 0.5, "compensation": 0.5, "location": 0.5, "growth": g}
        fit = build_fit(job(str(i)), comps, RankConfig(), [], None)
        act = FeedbackAction.SAVED if g > 0.5 else FeedbackAction.SKIPPED
        store.record(Feedback(job_id=str(i), action=act, at=NOW), fit)
    res = rank_jobs([job("a"), job("b")], PROFILE, ITEMS, fake({}), HashEmbedder(), feedback=store)
    p = res.personalization
    assert p.status == "active" and p.n_positive == 10 and p.n_negative == 10
    assert p.coefficients["growth"] > 1.0 and 0 < p.learned_weight <= 0.8
    assert res.ranked[0].learned_score is not None


def test_learned_weight_grows_with_feedback():
    from careerops.rank import Personalizer
    from careerops.rank.feedback import FeedbackRow

    def rows(n):
        out = []
        for i in range(n):
            g = 0.9 if i % 2 == 0 else 0.1
            comps = {"skills": 0.5, "seniority": 0.5, "compensation": 0.5, "location": 0.5, "growth": g}
            act = FeedbackAction.SAVED if i % 2 == 0 else FeedbackAction.SKIPPED
            out.append(FeedbackRow(feedback=Feedback(job_id=str(i), action=act, at=NOW),
                                   components=comps, rules_score=0.5, scorer_version="t"))
        return out

    small = Personalizer.from_feedback(rows(10)).info.learned_weight
    big = Personalizer.from_feedback(rows(60)).info.learned_weight
    assert 0 < small < big <= 0.8


def test_only_one_class_stays_insufficient():
    from careerops.rank import Personalizer
    from careerops.rank.feedback import FeedbackRow
    comps = {"skills": 0.5, "seniority": 0.5, "compensation": 0.5, "location": 0.5, "growth": 0.5}
    rows = [FeedbackRow(feedback=Feedback(job_id=str(i), action=FeedbackAction.SAVED, at=NOW),
                        components=comps, rules_score=0.5, scorer_version="t") for i in range(30)]
    assert Personalizer.from_feedback(rows).info.status == "insufficient_data"

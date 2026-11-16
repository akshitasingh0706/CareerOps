"""Seeded fabrication evals for the grounding check.

Numeric fabrications must be caught by code alone: the fake judge always says "supported".
Non-numeric fabrications (tools, seniority, scope) are caught only through the judge, and
the judge responses here are scripted, so those cases test the wiring, not judge quality.
"""
from collections import defaultdict
from datetime import UTC, datetime

import pytest

from careerops.domain import ResumeBullet, ResumeVersion
from careerops.exp import check_grounding_code, verify
from careerops.llm import FakeLLM
from tests.exp.conftest import make_item

ITEMS = {
    "lat": make_item("lat", "Cut p95 latency on the checkout API by 40% using Python and Postgres",
                     metrics=["cut p95 latency 40%"], skills=["Python", "Postgres"],
                     title="Software Engineer"),
    "mig": make_item("mig", "Migrated 12 services to AWS and saved $250k per year",
                     metrics=["12 services", "$250k annual savings"], skills=["AWS"]),
    "mgr": make_item("mgr", "Mentored 2 junior engineers on the payments team", title="Engineer"),
}

# (kind, bullet text, cited ids, judge unsupported claims, expected verified)
NUMERIC = "numeric"
CASES = [
    ("invented_metric", "Cut checkout latency 40% and raised conversion 15%", ["lat"], [], False),
    ("invented_metric", "Migrated 12 services to AWS, cutting incidents by 30%", ["mig"], [], False),
    ("invented_metric", "Mentored 2 engineers and grew team revenue 25%", ["mgr"], [], False),
    ("inflated_number", "Cut checkout API p95 latency by 60%", ["lat"], [], False),
    ("inflated_number", "Migrated 20 services to AWS", ["mig"], [], False),
    ("inflated_number", "Saved $2.5M per year by moving services to AWS", ["mig"], [], False),
    ("inflated_number", "Saved $250M per year on AWS", ["mig"], [], False),
    ("inflated_number", "Mentored 5 junior engineers", ["mgr"], [], False),
    ("invented_identifier", "Cut p99 latency using Python", ["lat"], [], False),
    ("invented_tool", "Cut latency 40% using Python, Postgres and Kafka", ["lat"],
     ["Kafka is not in the sources"], False),
    ("invented_tool", "Migrated 12 services to AWS using Terraform", ["mig"],
     ["Terraform is not in the sources"], False),
    ("changed_seniority", "Led the payments team and cut latency 40%", ["lat"],
     ["'led the team' is not supported"], False),
    ("changed_seniority", "Mentored 2 junior engineers as engineering manager", ["mgr"],
     ["manager title not supported"], False),
    ("invented_scope", "Migrated 12 services to AWS across the whole company", ["mig"],
     ["company-wide scope not supported"], False),
    ("bad_citation", "Cut latency 40%", ["nope"], [], False),
    ("bad_citation", "Cut latency 40%", [], [], False),
    ("valid_rewrite", "Reduced p95 latency of the checkout API by 40% with Python and Postgres",
     ["lat"], [], True),
    ("valid_rewrite", "Moved 12 services to AWS, saving $250k a year", ["mig"], [], True),
    ("valid_rewrite", "Mentored two junior engineers on payments", ["mgr"], [], True),
    ("valid_rewrite", "Cut checkout latency 40 percent; migrated 12 services to AWS",
     ["lat", "mig"], [], True),
]
NUMERIC_KINDS = {"invented_metric", "inflated_number", "invented_identifier"}


def _resume(text, ids):
    return ResumeVersion(id="r", job_id="j", created_at=datetime.now(UTC),
                         bullets=[ResumeBullet(text=text, source_item_ids=ids)])


def _judge(claims):
    return FakeLLM(structured={"JudgeResult": {"judgements": [
        {"bullet_index": 0, "supported": not claims, "unsupported_claims": claims}]}})


def test_case_count():
    assert len(CASES) >= 15


@pytest.mark.parametrize("kind,text,ids,claims,expected", CASES)
def test_verify_case(kind, text, ids, claims, expected):
    out = verify(_resume(text, ids), ITEMS, _judge(claims))
    assert out.verified is expected
    assert bool(out.verification_notes) is (not expected)


@pytest.mark.parametrize("kind,text,ids,claims,expected", CASES)
def test_numeric_and_citation_failures_caught_without_llm(kind, text, ids, claims, expected):
    """With a judge that approves everything, code alone must fail numeric and citation cases."""
    if kind not in NUMERIC_KINDS | {"bad_citation"}:
        pytest.skip("not detectable by code")
    out = verify(_resume(text, ids), ITEMS, _judge([]))
    assert out.verified is False
    assert check_grounding_code(_resume(text, ids), ITEMS)


def test_unapproved_item_is_rejected():
    out = verify(_resume("Cut latency 40%", ["lat"]), ITEMS, _judge([]), approved_ids={"mig"})
    assert not out.verified and "not approved" in out.verification_notes[0]


def test_catch_rate_report(capsys):
    """Print catch rate per case type. Run with -s to see it."""
    stats = defaultdict(lambda: [0, 0])  # kind -> [caught_or_passed, total]
    code_only = defaultdict(lambda: [0, 0])
    for kind, text, ids, claims, expected in CASES:
        out = verify(_resume(text, ids), ITEMS, _judge(claims))
        stats[kind][1] += 1
        stats[kind][0] += int(out.verified is expected)
        if not expected:
            code_only[kind][1] += 1
            code_only[kind][0] += int(bool(check_grounding_code(_resume(text, ids), ITEMS)))
    lines = [f"{'case type':22} {'correct (full)':>15} {'caught by code only':>20}"]
    for kind, (ok, n) in stats.items():
        c = code_only.get(kind)
        lines.append(f"{kind:22} {ok:>7}/{n:<7} {(f'{c[0]}/{c[1]}' if c else 'n/a'):>20}")
    with capsys.disabled():
        print("\n" + "\n".join(lines))
    for kind, (ok, n) in stats.items():
        assert ok == n, kind
    for kind in NUMERIC_KINDS:
        assert code_only[kind][0] == code_only[kind][1]

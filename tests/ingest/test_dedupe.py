from datetime import UTC, datetime

import pytest

from careerops.db import connect
from careerops.domain import Job, Source
from careerops.ingest.common import make_job_id, slugify
from careerops.ingest.dedupe import (
    SamePosting,
    are_duplicates,
    dedupe_jobs,
    dedupe_store,
    normalize_company,
    normalize_title,
)
from careerops.ingest.store import get_job, init_db, list_jobs, upsert_job
from careerops.llm import FakeLLM, HashEmbedder

EMB = HashEmbedder()


def mk(company, title, loc=None, desc="", source=Source.GREENHOUSE, sid=None):
    sid = sid or f"{title}-{loc}-{source.value}"
    return Job(id=make_job_id(source, sid), source=source, source_id=sid, url="u",
               company_id=slugify(company), company_name=company, title=title, location=loc,
               description=desc or f"{title} at {company}", fetched_at=datetime.now(UTC))


def test_normalize():
    assert normalize_company("Acme, Inc.") == "acme"
    assert normalize_company("ACME AI LLC") == "acme ai"
    assert normalize_title("Sr. ML Engineer (Remote)") == "senior machine learning engineer"
    assert normalize_title("Software Engineer II") == "software engineer 2"


class Boom:
    def structured(self, *a, **k):
        raise AssertionError("LLM must not be called")

    def complete(self, *a, **k):
        raise AssertionError("LLM must not be called")


def test_clear_cases_never_call_llm():
    a = mk("Acme", "Senior ML Engineer", "San Francisco, CA", "Train models and ship ranking systems.")
    b = mk("Acme Inc", "Senior Machine Learning Engineer", "SF", "Train models and ship ranking systems.",
           Source.EMAIL_ALERT)
    c = mk("Acme", "Office Manager", "San Francisco, CA", "Run the office and order snacks.")
    assert are_duplicates(a, b, EMB, Boom())
    assert not are_duplicates(a, c, EMB, Boom())


def test_borderline_goes_to_judge_and_answer_is_used():
    a = mk("Acme", "Machine Learning Engineer", "Berlin", "Work on search ranking.")
    b = mk("Acme", "Machine Learning Engineer, Search", "Berlin", "Work on search ranking and retrieval.",
           Source.LEVER)
    llm_yes = FakeLLM(structured={"SamePosting": SamePosting(same=True, reason="same team")})
    llm_no = FakeLLM(structured={"SamePosting": SamePosting(same=False, reason="different team")})
    assert are_duplicates(a, b, EMB, llm_yes, high=0.99, low=0.1)
    assert not are_duplicates(a, b, EMB, llm_no, high=0.99, low=0.1)
    assert not are_duplicates(a, b, EMB, None, high=0.99, low=0.1)
    assert len(llm_yes.calls) == 1


def test_dedupe_jobs_canonical_prefers_ats():
    ats = mk("Acme", "Senior Data Engineer", "Seattle, WA", "Build pipelines with Spark and Airflow "
             "for the analytics platform. Full description here.")
    alert = mk("Acme", "Sr. Data Engineer", "Seattle", "Build pipelines with Spark and Airflow.",
               Source.EMAIL_ALERT)
    other = mk("Acme", "Recruiter", "Seattle, WA", "Hire people.")
    out = dedupe_jobs([alert, other, ats], EMB, Boom())
    assert out[alert.id] == out[ats.id] == ats.id
    assert out[other.id] == other.id


def test_different_levels_and_locations_split():
    a = mk("Acme", "Software Engineer", "Austin, TX", "Build the product backend in Go.")
    b = mk("Acme", "Staff Software Engineer", "Austin, TX", "Build the product backend in Go.")
    c = mk("Acme", "Software Engineer", "London, UK", "Build the product backend in Go.")
    llm = FakeLLM(structured={"SamePosting": SamePosting(same=False, reason="other city")})
    out = dedupe_jobs([a, b, c], EMB, llm)
    assert len(set(out.values())) == 3
    assert len(llm.calls) == 2  # a-c and b-c reach the judge; a-b is split in code (level clash)


def test_dedupe_store_sets_canonical_and_caches_judgments():
    conn = connect(":memory:")
    init_db(conn)
    a = mk("Acme", "Machine Learning Engineer", "Berlin", "Work on search ranking.")
    b = mk("Acme", "Machine Learning Engineer, Search", "Berlin", "Work on search ranking.", Source.LEVER)
    for j in (a, b):
        upsert_job(conn, j)
    llm = FakeLLM(structured={"SamePosting": SamePosting(same=True, reason="same")})
    dedupe_store(conn, EMB, llm, high=0.99, low=0.1)
    ids = {j.canonical_id for j in list_jobs(conn)}
    assert len(ids) == 1
    assert get_job(conn, a.id).canonical_id == get_job(conn, b.id).canonical_id
    n = len(llm.calls)
    dedupe_store(conn, EMB, llm, high=0.99, low=0.1)
    assert len(llm.calls) == n  # judgment came from the cache


@pytest.mark.parametrize("bad", [None])
def test_no_llm_borderline_is_split(bad):
    a = mk("Acme", "Machine Learning Engineer", "Berlin", "Work on search ranking.")
    b = mk("Acme", "Machine Learning Engineer, Search", "Berlin", "Work on search ranking.", Source.LEVER)
    out = dedupe_jobs([a, b], EMB, bad, high=0.99, low=0.1)
    assert out[a.id] != out[b.id]

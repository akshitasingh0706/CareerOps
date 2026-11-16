from datetime import UTC, datetime

from careerops.domain import Job, Seniority, Source
from careerops.ingest.extract import JobFields, ManualJob, extract_fields, job_from_text
from careerops.llm import FakeLLM


def make_job(**kw):
    base = {"id": "j1", "source": Source.GREENHOUSE, "source_id": "1", "url": "u",
            "company_id": "acme", "company_name": "Acme", "title": "ML Engineer",
            "description": "Build models. Pay 100-150k.", "fetched_at": datetime.now(UTC)}
    return Job(**{**base, **kw})


def test_fills_missing_fields():
    llm = FakeLLM(structured={"JobFields": JobFields(
        seniority=Seniority.SENIOR, remote=True, salary_min=100000, salary_max=150000,
        salary_currency="usd")})
    out = extract_fields(make_job(), llm)
    assert out.seniority == Seniority.SENIOR and out.remote is True
    assert (out.salary_min, out.salary_max, out.salary_currency) == (100000, 150000, "USD")
    assert len(llm.calls) == 1


def test_no_call_when_nothing_missing():
    llm = FakeLLM()
    job = make_job(seniority=Seniority.MID, remote=False, salary_min=1, salary_max=2)
    assert extract_fields(job, llm) is job
    assert llm.calls == []


def test_does_not_overwrite_source_fields():
    llm = FakeLLM(structured={"JobFields": JobFields(
        seniority=Seniority.STAFF, remote=True, salary_min=1, salary_max=2, salary_currency="EUR")})
    job = make_job(remote=False, salary_min=90000, salary_max=95000, salary_currency="USD")
    out = extract_fields(job, llm)
    assert out.remote is False
    assert (out.salary_min, out.salary_currency) == (90000, "USD")
    assert out.seniority == Seniority.STAFF
    assert "salary" not in llm.calls[0][1].split("Fields needed:")[1].split("\n")[0]


def test_rejects_inconsistent_salary():
    llm = FakeLLM(structured={"JobFields": {"salary_min": 200000, "salary_max": 100000}})
    assert extract_fields(make_job(), llm).salary_min is None


def test_job_from_text():
    llm = FakeLLM(structured={"ManualJob": ManualJob(
        title="Staff Engineer", company_name="Acme Corp", location="Remote",
        description="Lead things.", seniority=Seniority.STAFF, remote=True,
        salary_min=200000, salary_max=240000, salary_currency="USD")})
    j = job_from_text("raw text", url="https://x.com/job/1", llm=llm)
    assert j.source == Source.MANUAL and j.url == "https://x.com/job/1"
    assert j.company_id == "acme-corp" and j.seniority == Seniority.STAFF
    assert j.salary_max == 240000
    j2 = job_from_text("raw text", url="https://x.com/job/1", llm=llm)
    assert j.id == j2.id

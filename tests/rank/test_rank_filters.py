from datetime import UTC, datetime

from careerops.domain import Job, Seniority, Source, UserProfile
from careerops.rank import FilterConfig, apply_filters, is_unknown, passes_all
from careerops.rank import filters as F


def job(**kw) -> Job:
    base = {
        "id": "j", "source": Source.MANUAL, "source_id": "1", "url": "u", "company_id": "acme",
        "company_name": "Acme", "title": "Engineer", "description": "Python and SQL work",
        "fetched_at": datetime(2026, 1, 1, tzinfo=UTC), "seniority": Seniority.SENIOR,
    }
    base.update(kw)
    return Job(**base)


def by_name(results):
    return {r.name: r for r in results}


def test_remote_job_passes_location_when_remote_ok():
    r = F.location_filter(job(remote=True), UserProfile(locations=["Boston"], remote_ok=True))
    assert r.passed and "remote" in r.detail


def test_remote_job_fails_when_remote_not_ok():
    r = F.location_filter(job(remote=True), UserProfile(locations=["Boston"], remote_ok=False))
    assert not r.passed and "does not accept remote" in r.detail


def test_location_match_is_case_insensitive_substring():
    p = UserProfile(locations=["new york"], remote_ok=False)
    assert F.location_filter(job(location="New York, NY", remote=False), p).passed
    r = F.location_filter(job(location="Austin, TX", remote=False), p)
    assert not r.passed and "Austin" in r.detail


def test_missing_location_is_unknown_by_default():
    p = UserProfile(locations=["Boston"])
    r = F.location_filter(job(location=None, remote=None), p)
    assert r.passed and is_unknown(r)
    r = F.location_filter(job(location=None, remote=None), p, FilterConfig(location_missing="fail"))
    assert not r.passed
    r = F.location_filter(job(location=None, remote=None), p, FilterConfig(location_missing="pass"))
    assert r.passed and not is_unknown(r)


def test_remote_in_location_text_counts_as_remote():
    r = F.location_filter(job(location="Remote - US", remote=None), UserProfile(locations=["Boston"]))
    assert r.passed and not is_unknown(r)


def test_seniority_range():
    p = UserProfile(min_seniority=Seniority.MID, max_seniority=Seniority.SENIOR)
    assert F.seniority_filter(job(seniority=Seniority.SENIOR), p).passed
    r = F.seniority_filter(job(seniority=Seniority.STAFF), p)
    assert not r.passed and "outside" in r.detail
    r = F.seniority_filter(job(seniority=Seniority.UNKNOWN), p)
    assert r.passed and is_unknown(r)


def test_seniority_open_ended_range():
    p = UserProfile(min_seniority=Seniority.SENIOR)
    assert F.seniority_filter(job(seniority=Seniority.PRINCIPAL), p).passed
    assert not F.seniority_filter(job(seniority=Seniority.JUNIOR), p).passed


def test_compensation_missing_salary_is_unknown_not_pass_or_fail():
    p = UserProfile(min_salary=150_000)
    r = F.compensation_filter(job(), p)
    assert r.passed and is_unknown(r)
    assert not F.compensation_filter(job(), p, FilterConfig(compensation_missing="fail")).passed
    quiet = F.compensation_filter(job(), p, FilterConfig(compensation_missing="pass"))
    assert quiet.passed and not is_unknown(quiet)


def test_compensation_uses_top_of_range_and_checks_currency():
    p = UserProfile(min_salary=150_000)
    assert F.compensation_filter(job(salary_min=120_000, salary_max=160_000), p).passed
    r = F.compensation_filter(job(salary_min=100_000, salary_max=140_000), p)
    assert not r.passed and "140000" in r.detail
    assert F.compensation_filter(job(salary_min=160_000), p).passed
    r = F.compensation_filter(job(salary_max=90_000, salary_currency="EUR"), p)
    assert is_unknown(r)


def test_no_minimum_salary_always_passes():
    assert F.compensation_filter(job(), UserProfile()).passed


def test_avoid_companies():
    p = UserProfile(avoid_companies=["ACME"])
    r = F.avoid_companies_filter(job(), p)
    assert not r.passed and "Acme" in r.detail
    assert F.avoid_companies_filter(job(company_name="Other", company_id="other"), p).passed


def test_must_have_skills_whole_word():
    p = UserProfile(must_have_skills=["python", "C++", "go"])
    r = F.must_have_skills_filter(job(description="Python and golang"), p)
    assert not r.passed and "C++" in r.detail and "go" in r.detail
    r = F.must_have_skills_filter(job(description="python, c++ and go."), p)
    assert r.passed


def test_apply_filters_returns_all_five_and_passes_all():
    res = apply_filters(job(remote=True), UserProfile())
    assert [r.name for r in res] == [
        "location", "seniority", "compensation", "avoid_companies", "must_have_skills",
    ]
    assert passes_all(res)
    assert not passes_all(apply_filters(job(), UserProfile(avoid_companies=["acme"])))
    assert "avoid_companies" in by_name(apply_filters(job(), UserProfile(avoid_companies=["acme"])))

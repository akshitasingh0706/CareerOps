import hashlib

import pytest

from careerops.domain import Company, Source
from careerops.ingest.common import html_to_text, make_job_id
from careerops.ingest.connectors import (
    fetch_ashby,
    fetch_company_jobs,
    fetch_greenhouse,
    fetch_lever,
)


def test_job_id_is_sha1_of_source_and_source_id():
    assert make_job_id(Source.LEVER, "abc") == hashlib.sha1(b"leverabc").hexdigest()


def test_html_to_text():
    t = html_to_text("<p>Hi &amp; bye</p><ul><li>One</li><li>Two</li></ul><script>x()</script>")
    assert t == "Hi & bye\n\n- One\n- Two"


def test_html_to_text_handles_escaped_html():
    assert html_to_text("&lt;p&gt;Hello&lt;/p&gt;") == "Hello"


def test_greenhouse(mock_client):
    pairs = fetch_greenhouse("acme", "Acme AI", client=mock_client)
    assert "boards-api.greenhouse.io/v1/boards/acme/jobs?content=true" in mock_client.state["requests"][0]
    jobs = [j for j, _ in pairs]
    assert len(jobs) == 2
    j = jobs[0]
    assert j.source == Source.GREENHOUSE and j.source_id == "4012345"
    assert j.id == make_job_id(Source.GREENHOUSE, "4012345")
    assert j.title == "Senior Machine Learning Engineer"
    assert "<" not in j.description and "Build ranking models & ship them." in j.description
    assert j.location == "San Francisco, CA"
    assert j.company_id == "acme-ai"
    assert j.posted_at is not None
    assert jobs[1].remote is True
    assert pairs[0][1]["id"] == 4012345  # raw kept


def test_lever(mock_client):
    jobs = [j for j, _ in fetch_lever("beta", "Beta", client=mock_client)]
    assert len(jobs) == 2
    a, b = jobs
    assert a.url.endswith("a1b2c3d4-0001")
    assert "Design APIs" in a.description and "equal opportunity" in a.description
    assert (a.salary_min, a.salary_max, a.salary_currency) == (200000, 250000, "USD")
    assert a.remote is False and b.remote is True
    assert a.posted_at.year == 2025


def test_ashby_skips_unlisted(mock_client):
    jobs = [j for j, _ in fetch_ashby("gamma", "Gamma", client=mock_client)]
    assert [j.title for j in jobs] == ["Sr. ML Engineer", "Engineering Manager"]
    assert jobs[0].salary_currency == "GBP" and jobs[0].salary_min == 120000
    assert jobs[1].remote is True
    assert jobs[1].description == "Lead a team of eight engineers."


def test_fetch_company_jobs_requires_ats(mock_client):
    with pytest.raises(ValueError):
        fetch_company_jobs(Company(id="x", name="X"), mock_client)

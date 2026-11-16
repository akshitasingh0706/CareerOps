from pathlib import Path

from careerops.domain import Source
from careerops.ingest.alerts import parse_alert_email, read_eml

FIX = Path(__file__).parent / "fixtures"


def read(name):
    return (FIX / name).read_text()


def test_linkedin_html():
    jobs = parse_alert_email(read("linkedin_alert.html"))
    assert len(jobs) == 2
    j = jobs[0]
    assert j.source == Source.EMAIL_ALERT
    assert j.title == "Senior Machine Learning Engineer"
    assert j.company_name == "Acme AI"
    assert j.location == "San Francisco, CA (Hybrid)"
    assert j.url == "https://www.linkedin.com/jobs/view/3912345678"
    assert jobs[1].title == "Data Scientist" and jobs[1].remote is True


def test_wellfound_html():
    jobs = parse_alert_email(read("wellfound_alert.html"))
    assert [j.title for j in jobs] == ["Founding ML Engineer", "Backend Engineer"]
    assert jobs[0].company_name == "Gamma Labs" and jobs[0].location == "Remote"
    assert jobs[1].url == "https://wellfound.com/jobs/3344556-backend-engineer"
    assert "payment infrastructure" in jobs[1].description


def test_indeed_html():
    jobs = parse_alert_email(read("indeed_alert.html"))
    assert len(jobs) == 2
    assert jobs[0].url == "https://www.indeed.com/viewjob?jk=5f6a7b8c9d0e1f20"
    assert jobs[0].company_name == "Epsilon Corp" and jobs[0].location == "Seattle, WA"
    assert "Spark" in jobs[0].description
    assert jobs[1].title == "Sr. Data Engineer"


def test_plain_text():
    jobs = parse_alert_email(read("plain_alert.txt"))
    assert [(j.title, j.company_name, j.location) for j in jobs] == [
        ("Software Engineer, Backend", "Orbit Systems", "Austin, TX"),
        ("Platform Engineer", "Delta Health", "Remote"),
    ]


def test_eml_file_and_raw_eml_match():
    from_file = read_eml(FIX / "alert.eml")
    from_raw = parse_alert_email(read("alert.eml"))
    assert len(from_file) == 1
    assert from_file[0].id == from_raw[0].id
    assert from_file[0].url == "https://www.indeed.com/viewjob?jk=5f6a7b8c9d0e1f20"


def test_ids_stable_and_unique():
    a = parse_alert_email(read("linkedin_alert.html"))
    b = parse_alert_email(read("linkedin_alert.html"))
    assert [j.id for j in a] == [j.id for j in b]
    assert len({j.id for j in a}) == 2


def test_no_jobs():
    assert parse_alert_email("<html><body><p>Hello</p></body></html>") == []

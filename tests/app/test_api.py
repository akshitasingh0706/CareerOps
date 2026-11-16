"""End-to-end API test on a temporary database. No network."""
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from careerops import ingest, service
from careerops.api import app
from careerops.domain import Company, Job, Source


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREEROPS_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("CAREEROPS_PROFILE", str(tmp_path / "profile.json"))
    with service.session() as (conn, _):
        ingest.add_company(conn, Company(id="acme", name="Acme", watched=False))
        for i, title in enumerate(["ML Engineer", "Sales Manager"]):
            job = Job(
                id=f"job{i}", source=Source.MANUAL, source_id=str(i), url=f"https://x.test/{i}",
                company_id="acme", company_name="Acme", title=title,
                description="Build machine learning systems in Python." if i == 0 else "Sell things.",
                fetched_at=datetime(2026, 1, 1),
            )
            ingest.upsert_job(conn, job, {})
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_profile_round_trip(client):
    p = client.get("/profile").json()
    p["target_titles"] = ["ML Engineer"]
    assert client.put("/profile", json=p).status_code == 200
    assert client.get("/profile").json()["target_titles"] == ["ML Engineer"]


def test_rank_save_and_advance(client):
    ranked = client.get("/jobs/ranked").json()
    assert {j["id"] for j in ranked["ranked"]} == {"job0", "job1"}

    saved = client.post("/jobs/job0/save").json()
    assert saved["stage"] == "saved"
    # saving twice returns the same application
    assert client.post("/jobs/job0/save").json()["id"] == saved["id"]
    assert client.get("/pipeline").json()["saved"] == 1

    ok = client.post(f"/applications/{saved['id']}/advance", json={"stage": "applied"})
    assert ok.json()["stage"] == "applied"
    bad = client.post(f"/applications/{saved['id']}/advance", json={"stage": "saved"})
    assert bad.status_code == 400


def test_skip_hides_job_and_logs_feedback(client):
    assert client.post("/jobs/job1/skip", json={}).status_code == 200
    ids = {j["id"] for j in client.get("/jobs/ranked").json()["ranked"]}
    assert ids == {"job0"}


def test_unknown_job_is_404(client):
    assert client.post("/jobs/nope/save").status_code == 404

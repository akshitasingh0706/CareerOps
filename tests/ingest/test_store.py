import httpx
import pytest

from careerops.db import connect
from careerops.domain import Company, Seniority, Source
from careerops.ingest.connectors import fetch_company_jobs, parse_greenhouse
from careerops.ingest.extract import JobFields
from careerops.ingest.store import (
    add_company,
    add_to_watchlist,
    get_job,
    get_raw,
    init_db,
    list_companies,
    list_jobs,
    mark_closed,
    poll_watchlist,
    upsert_job,
)
from careerops.llm import FakeLLM

from .conftest import load

ACME = Company(id="acme-ai", name="Acme AI", ats=Source.GREENHOUSE, board_token="acme")


@pytest.fixture
def conn():
    c = connect(":memory:")
    init_db(c)
    return c


def test_upsert_keeps_raw_and_extracted_fields(conn):
    (job, raw), _ = parse_greenhouse(load("greenhouse_acme.json"), ACME)[:2]
    assert upsert_job(conn, job, raw) is True
    stored = get_job(conn, job.id)
    stored.seniority = Seniority.SENIOR
    from careerops.ingest.store import save_job
    save_job(conn, stored)
    assert upsert_job(conn, job, raw) is False  # same job fetched again
    again = get_job(conn, job.id)
    assert again.seniority == Seniority.SENIOR
    assert get_raw(conn, job.id)["id"] == 4012345
    assert len(list_jobs(conn)) == 1


def test_watchlist_poll_new_and_closed(conn, mock_client):
    add_to_watchlist(conn, ACME)
    add_company(conn, Company(id="other", name="Other", ats=Source.LEVER, board_token="o"))
    assert [c.id for c in list_companies(conn, watched_only=True)] == ["acme-ai"]

    new = poll_watchlist(conn, client=mock_client)
    assert len(new) == 2
    assert poll_watchlist(conn, client=mock_client) == []

    # Next poll: one job disappears and a new one appears.
    payload = load("greenhouse_acme.json")
    payload["jobs"] = [payload["jobs"][0],
                       {**payload["jobs"][1], "id": 777, "title": "New Role",
                        "absolute_url": "https://boards.greenhouse.io/acme/jobs/777"}]

    def fetcher(company, client):
        return parse_greenhouse(payload, company)

    new = poll_watchlist(conn, fetcher=fetcher)
    assert [j.title for j in new] == ["New Role"]
    open_ids = {j.source_id for j in list_jobs(conn)}
    assert open_ids == {"4012345", "777"}
    closed = [j for j in list_jobs(conn, include_closed=True) if j.closed]
    assert [j.source_id for j in closed] == ["4012399"]


def test_failed_fetch_does_not_close_jobs(conn, mock_client):
    add_to_watchlist(conn, ACME)
    poll_watchlist(conn, client=mock_client)
    mock_client.state["override"]["boards-api.greenhouse.io"] = httpx.Response(500)
    assert poll_watchlist(conn, client=mock_client) == []
    assert len(list_jobs(conn)) == 2


def test_poll_extracts_for_new_jobs_only(conn, mock_client):
    add_to_watchlist(conn, ACME)
    llm = FakeLLM(structured={"JobFields": JobFields(seniority=Seniority.SENIOR)})
    new = poll_watchlist(conn, client=mock_client, llm=llm)
    assert all(j.seniority == Seniority.SENIOR for j in new)
    n_calls = len(llm.calls)
    poll_watchlist(conn, client=mock_client, llm=llm)
    assert len(llm.calls) == n_calls


def test_reopen_and_mark_closed(conn, mock_client):
    pairs = fetch_company_jobs(ACME, mock_client)
    for j, raw in pairs:
        upsert_job(conn, j, raw)
    assert mark_closed(conn, [pairs[0][0].id]) == 1
    assert len(list_jobs(conn)) == 1
    upsert_job(conn, *pairs[0])
    assert len(list_jobs(conn)) == 2

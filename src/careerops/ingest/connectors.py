"""ATS connectors: Greenhouse, Lever, Ashby. Each returns Job objects and the raw payloads.

The httpx client is injectable so tests can use recorded fixtures.
"""
from __future__ import annotations

from datetime import UTC, datetime

import httpx

from careerops.domain import Company, Job, Source
from careerops.ingest.common import html_to_text, make_job_id, slugify

GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
LEVER_URL = "https://api.lever.co/v0/postings/{token}?mode=json"
ASHBY_URL = "https://api.ashbyhq.com/posting-api/job-board/{token}"

RawJob = tuple[Job, dict]


def _now() -> datetime:
    return datetime.now(UTC)


def _ms_to_dt(ms: int | None) -> datetime | None:
    return datetime.fromtimestamp(ms / 1000, UTC) if ms else None


def _iso_to_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _remote_from_text(*values: str | None) -> bool | None:
    joined = " ".join(v for v in values if v).lower()
    if not joined:
        return None
    return True if "remote" in joined else None


def parse_greenhouse(payload: dict, company: Company, fetched_at: datetime | None = None) -> list[RawJob]:
    """Turn a Greenhouse board payload into (Job, raw) pairs."""
    now = fetched_at or _now()
    out: list[RawJob] = []
    for raw in payload.get("jobs", []):
        sid = str(raw["id"])
        loc = (raw.get("location") or {}).get("name")
        title = raw.get("title", "")
        out.append((
            Job(
                id=make_job_id(Source.GREENHOUSE, sid), source=Source.GREENHOUSE, source_id=sid,
                url=raw.get("absolute_url", ""), company_id=company.id,
                company_name=raw.get("company_name") or company.name, title=title, location=loc,
                remote=_remote_from_text(loc, title),
                description=html_to_text(raw.get("content")),
                posted_at=_iso_to_dt(raw.get("first_published") or raw.get("updated_at")),
                fetched_at=now,
            ),
            raw,
        ))
    return out


def parse_lever(payload: list[dict], company: Company, fetched_at: datetime | None = None) -> list[RawJob]:
    """Turn a Lever postings list into (Job, raw) pairs."""
    now = fetched_at or _now()
    out: list[RawJob] = []
    for raw in payload:
        cats = raw.get("categories") or {}
        loc = cats.get("location")
        desc = raw.get("descriptionPlain") or html_to_text(raw.get("description"))
        for lst in raw.get("lists") or []:
            desc += f"\n\n{lst.get('text', '')}\n{html_to_text(lst.get('content'))}"
        if raw.get("additionalPlain"):
            desc += "\n\n" + raw["additionalPlain"]
        remote = None
        if raw.get("workplaceType") == "remote":
            remote = True
        elif raw.get("workplaceType") in ("onsite", "hybrid"):
            remote = False
        else:
            remote = _remote_from_text(loc)
        sal = raw.get("salaryRange") or {}
        out.append((
            Job(
                id=make_job_id(Source.LEVER, raw["id"]), source=Source.LEVER, source_id=raw["id"],
                url=raw.get("hostedUrl", ""), company_id=company.id, company_name=company.name,
                title=raw.get("text", ""), location=loc, remote=remote, description=desc.strip(),
                posted_at=_ms_to_dt(raw.get("createdAt")), fetched_at=now,
                salary_min=int(sal["min"]) if sal.get("min") else None,
                salary_max=int(sal["max"]) if sal.get("max") else None,
                salary_currency=sal.get("currency"),
            ),
            raw,
        ))
    return out


def parse_ashby(payload: dict, company: Company, fetched_at: datetime | None = None) -> list[RawJob]:
    """Turn an Ashby job board payload into (Job, raw) pairs."""
    now = fetched_at or _now()
    out: list[RawJob] = []
    for raw in payload.get("jobs", []):
        if raw.get("isListed") is False:
            continue
        loc = raw.get("location")
        remote = raw.get("isRemote")
        if remote is None:
            remote = _remote_from_text(loc)
        comp = (raw.get("compensation") or {})
        smin = smax = cur = None
        for comp_summary in comp.get("summaryComponents") or []:
            if comp_summary.get("compensationType") == "Salary":
                smin, smax = comp_summary.get("minValue"), comp_summary.get("maxValue")
                cur = comp_summary.get("currencyCode")
        out.append((
            Job(
                id=make_job_id(Source.ASHBY, raw["id"]), source=Source.ASHBY, source_id=raw["id"],
                url=raw.get("jobUrl", ""), company_id=company.id, company_name=company.name,
                title=raw.get("title", ""), location=loc, remote=remote,
                description=raw.get("descriptionPlain") or html_to_text(raw.get("descriptionHtml")),
                posted_at=_iso_to_dt(raw.get("publishedAt")), fetched_at=now,
                salary_min=int(smin) if smin else None, salary_max=int(smax) if smax else None,
                salary_currency=cur,
            ),
            raw,
        ))
    return out


def _get_json(client: httpx.Client, url: str):
    resp = client.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_company_jobs(company: Company, client: httpx.Client | None = None) -> list[RawJob]:
    """Fetch all open jobs for a company with a known ATS and board token."""
    if not company.ats or not company.board_token:
        raise ValueError(f"company {company.id} has no ATS and board token")
    own = client is None
    client = client or httpx.Client(headers={"User-Agent": "CareerOps/0.1"})
    try:
        if company.ats == Source.GREENHOUSE:
            return parse_greenhouse(_get_json(client, GREENHOUSE_URL.format(token=company.board_token)), company)
        if company.ats == Source.LEVER:
            return parse_lever(_get_json(client, LEVER_URL.format(token=company.board_token)), company)
        if company.ats == Source.ASHBY:
            return parse_ashby(_get_json(client, ASHBY_URL.format(token=company.board_token)), company)
        raise ValueError(f"unsupported ATS: {company.ats}")
    finally:
        if own:
            client.close()


def fetch_greenhouse(token: str, company_name: str | None = None, client: httpx.Client | None = None) -> list[RawJob]:
    """Fetch jobs from a Greenhouse board."""
    c = Company(id=slugify(company_name or token), name=company_name or token, ats=Source.GREENHOUSE, board_token=token)
    return fetch_company_jobs(c, client)


def fetch_lever(token: str, company_name: str | None = None, client: httpx.Client | None = None) -> list[RawJob]:
    """Fetch jobs from a Lever board."""
    c = Company(id=slugify(company_name or token), name=company_name or token, ats=Source.LEVER, board_token=token)
    return fetch_company_jobs(c, client)


def fetch_ashby(token: str, company_name: str | None = None, client: httpx.Client | None = None) -> list[RawJob]:
    """Fetch jobs from an Ashby board."""
    c = Company(id=slugify(company_name or token), name=company_name or token, ats=Source.ASHBY, board_token=token)
    return fetch_company_jobs(c, client)

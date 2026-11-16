"""Ingestion: connectors, alert emails, manual paste, extraction, storage, dedupe."""
from careerops.ingest.alerts import parse_alert_email, read_eml
from careerops.ingest.connectors import (
    fetch_ashby,
    fetch_company_jobs,
    fetch_greenhouse,
    fetch_lever,
    parse_ashby,
    parse_greenhouse,
    parse_lever,
)
from careerops.ingest.dedupe import SamePosting, are_duplicates, dedupe_jobs, dedupe_store
from careerops.ingest.extract import JobFields, extract_fields, job_from_text
from careerops.ingest.store import (
    add_company,
    add_to_watchlist,
    get_job,
    ingest_jobs,
    init_db,
    list_companies,
    list_jobs,
    mark_closed,
    poll_watchlist,
    upsert_job,
)

__all__ = [
    "JobFields", "SamePosting", "add_company", "add_to_watchlist", "are_duplicates",
    "dedupe_jobs", "dedupe_store", "extract_fields", "fetch_ashby", "fetch_company_jobs",
    "fetch_greenhouse", "fetch_lever", "get_job", "ingest_jobs", "init_db", "job_from_text",
    "list_companies", "list_jobs", "mark_closed", "parse_alert_email", "parse_ashby",
    "parse_greenhouse", "parse_lever", "poll_watchlist", "read_eml", "upsert_job",
]

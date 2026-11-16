"""Structured extraction. The LLM reads the text. Code decides which fields to accept."""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from careerops.domain import Job, Seniority, Source
from careerops.ingest.common import make_job_id, slugify
from careerops.llm import CHEAP_MODEL, LLM

SYSTEM = (
    "You extract facts from job postings. Use only what the text says. "
    "If a fact is not stated, return null (or 'unknown' for seniority). Do not guess."
)


class JobFields(BaseModel):
    """Fields the LLM may fill from a description."""
    seniority: Seniority = Seniority.UNKNOWN
    remote: bool | None = Field(None, description="True if the role can be done remotely")
    salary_min: int | None = Field(None, description="Annual minimum, whole currency units")
    salary_max: int | None = Field(None, description="Annual maximum, whole currency units")
    salary_currency: str | None = Field(None, description="ISO code such as USD")


class ManualJob(JobFields):
    """What the LLM returns for a pasted job posting."""
    title: str
    company_name: str
    location: str | None = None
    description: str = Field(description="The job description as plain text")


def missing_fields(job: Job) -> set[str]:
    """Names of extractable fields the source did not provide."""
    missing = set()
    if job.seniority == Seniority.UNKNOWN:
        missing.add("seniority")
    if job.remote is None:
        missing.add("remote")
    if job.salary_min is None and job.salary_max is None:
        missing.add("salary")
    return missing


def _apply_salary(job: Job, f: JobFields) -> None:
    lo, hi = f.salary_min, f.salary_max
    if lo is not None and lo <= 0:
        lo = None
    if hi is not None and hi <= 0:
        hi = None
    if lo is not None and hi is not None and lo > hi:
        return  # inconsistent output, drop it
    if lo is None and hi is None:
        return
    job.salary_min, job.salary_max = lo, hi
    job.salary_currency = (f.salary_currency or "").upper() or None


def extract_fields(job: Job, llm: LLM, *, model: str | None = CHEAP_MODEL) -> Job:
    """Fill missing seniority, remote and salary from the description.

    Does nothing (and makes no LLM call) when the source already gave all of them.
    Fields the source already had are never overwritten.
    """
    missing = missing_fields(job)
    if not missing or not job.description.strip():
        return job
    prompt = (
        f"Title: {job.title}\nLocation: {job.location or 'n/a'}\n"
        f"Fields needed: {', '.join(sorted(missing))}\n\nDescription:\n{job.description[:8000]}"
    )
    f = llm.structured(JobFields, SYSTEM, prompt, model=model)
    out = job.model_copy()
    if "seniority" in missing:
        out.seniority = f.seniority
    if "remote" in missing:
        out.remote = f.remote
    if "salary" in missing:
        _apply_salary(out, f)
    return out


def job_from_text(text: str, url: str | None = None, llm: LLM | None = None,
                  *, model: str | None = None) -> Job:
    """Build a Job from pasted posting text. The LLM extracts every field."""
    if llm is None:
        raise ValueError("an llm is required")
    prompt = f"URL: {url or 'n/a'}\n\nPosting text:\n{text[:12000]}"
    m = llm.structured(ManualJob, SYSTEM, prompt, model=model)
    source_id = hashlib.sha1((url or text).encode()).hexdigest()[:16]
    job = Job(
        id=make_job_id(Source.MANUAL, source_id), source=Source.MANUAL, source_id=source_id,
        url=url or "", company_id=slugify(m.company_name), company_name=m.company_name,
        title=m.title, location=m.location, description=m.description or text,
        fetched_at=datetime.now(UTC),
    )
    job.seniority, job.remote = m.seniority, m.remote
    _apply_salary(job, m)
    return job

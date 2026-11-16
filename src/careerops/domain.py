"""Shared domain models. This file is the contract between modules.

Change it only when a module cannot work without the change, and keep changes additive.
Each module owns its own tables and persistence code (see db.py) and imports these types.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class Source(str, Enum):
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    CAREER_PAGE = "career_page"
    MANUAL = "manual"          # pasted by the user
    EMAIL_ALERT = "email_alert"  # forwarded alert emails
    AGGREGATOR = "aggregator"  # licensed aggregator APIs (Adzuna, JSearch, ...)
    BROWSER_CAPTURE = "browser_capture"  # job the user captured while viewing it


class Seniority(str, Enum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    MANAGER = "manager"
    UNKNOWN = "unknown"


class Company(BaseModel):
    id: str  # slug, e.g. "airbnb"
    name: str
    website: str | None = None
    ats: Source | None = None
    board_token: str | None = None  # Greenhouse/Lever/Ashby board identifier
    watched: bool = False


class Job(BaseModel):
    id: str  # stable: sha1(source + source_id)
    source: Source
    source_id: str
    url: str
    company_id: str
    company_name: str
    title: str
    location: str | None = None
    remote: bool | None = None
    description: str  # plain text
    posted_at: datetime | None = None
    fetched_at: datetime
    salary_min: int | None = None
    salary_max: int | None = None
    salary_currency: str | None = None
    seniority: Seniority = Seniority.UNKNOWN
    canonical_id: str | None = None  # set by dedupe: id of the cluster's representative job
    closed: bool = False


class ExperienceKind(str, Enum):
    ACCOMPLISHMENT = "accomplishment"
    STAR_STORY = "star_story"
    PROJECT = "project"
    SKILL = "skill"


class ExperienceItem(BaseModel):
    """One verified fact about the user's history. Resume bullets must trace back to these."""
    id: str
    kind: ExperienceKind
    title: str
    employer: str | None = None
    start: date | None = None
    end: date | None = None
    text: str  # the verified statement, in the user's words
    metrics: list[str] = Field(default_factory=list)  # e.g. "cut p95 latency 40%"
    skills: list[str] = Field(default_factory=list)
    situation: str | None = None
    task: str | None = None
    action: str | None = None
    result: str | None = None


class ResumeBullet(BaseModel):
    text: str
    source_item_ids: list[str]  # must be non-empty; every claim is grounded


class ResumeVersion(BaseModel):
    id: str
    job_id: str
    created_at: datetime
    bullets: list[ResumeBullet]
    verified: bool = False  # True only after the grounding check passes
    verification_notes: list[str] = Field(default_factory=list)
    model: str | None = None


class Stage(str, Enum):
    SAVED = "saved"
    PREPARING = "preparing"
    APPLIED = "applied"
    SCREEN = "screen"
    INTERVIEW = "interview"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class Application(BaseModel):
    id: str
    job_id: str
    stage: Stage = Stage.SAVED
    resume_version_id: str | None = None
    submitted_at: datetime | None = None
    submission_channel: str | None = None  # "company site", "referral", "recruiter email"
    deadline: date | None = None
    notes: str = ""
    updated_at: datetime


class Contact(BaseModel):
    id: str
    name: str
    role: str | None = None
    company_id: str | None = None
    email: str | None = None
    linkedin_url: str | None = None
    application_ids: list[str] = Field(default_factory=list)


class FollowUp(BaseModel):
    id: str
    application_id: str
    due_on: date
    note: str
    contact_id: str | None = None
    done: bool = False


class FilterResult(BaseModel):
    name: str  # "location", "seniority", "compensation", ...
    passed: bool
    detail: str


class FitScore(BaseModel):
    job_id: str
    overall: float  # 0..1, computed in code from components, never by the LLM
    components: dict[str, float]  # "skills", "seniority", "compensation", "location", "growth"
    filters: list[FilterResult]
    explanation: str  # LLM-written, cites job and profile facts
    evidence: list[str] = Field(default_factory=list)  # requirement -> experience item ids
    scorer_version: str


class FeedbackAction(str, Enum):
    SAVED = "saved"
    SKIPPED = "skipped"
    APPLIED = "applied"
    INTERVIEWED = "interviewed"
    REJECTED_BY_ME = "rejected_by_me"


class Feedback(BaseModel):
    job_id: str
    action: FeedbackAction
    at: datetime
    reason: str | None = None  # free text, optional


class UserProfile(BaseModel):
    """What the user wants. Hard filters are enforced in code."""
    target_titles: list[str] = Field(default_factory=list)
    min_seniority: Seniority = Seniority.UNKNOWN
    max_seniority: Seniority = Seniority.UNKNOWN
    locations: list[str] = Field(default_factory=list)
    remote_ok: bool = True
    min_salary: int | None = None
    salary_currency: str = "USD"
    must_have_skills: list[str] = Field(default_factory=list)
    avoid_companies: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)  # free text, used for growth/fit

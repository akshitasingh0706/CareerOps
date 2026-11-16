"""Requirement coverage report. Gaps are reported as gaps and never filled."""
from __future__ import annotations

from pydantic import BaseModel, Field

from careerops.domain import Job, ResumeVersion
from careerops.llm import LLM

SYSTEM = (
    "List the concrete requirements in the job posting (skills, experience, responsibilities). "
    "For each, say which resume bullets cover it. A bullet covers a requirement only if it "
    "clearly shows it. Quote a short evidence phrase copied exactly from the bullet. If no "
    "bullet covers it, leave bullet_indexes empty. Do not write new resume text."
)


class RequirementMatch(BaseModel):
    requirement: str
    bullet_indexes: list[int] = Field(default_factory=list)  # 0-based
    evidence: str | None = None


class RequirementMatches(BaseModel):
    requirements: list[RequirementMatch]


class RequirementCoverage(BaseModel):
    requirement: str
    covered: bool
    bullet_indexes: list[int]
    evidence: str | None = None


class CoverageReport(BaseModel):
    requirements: list[RequirementCoverage]
    gaps: list[str]
    coverage_ratio: float  # computed in code


def coverage_report(job: Job, resume: ResumeVersion, llm: LLM) -> CoverageReport:
    """Report which job requirements the resume covers.

    A requirement counts as covered only if the LLM cites real bullet indexes and its evidence
    phrase appears in one of those bullets. Everything else is a gap.
    """
    bullets = "\n".join(f"{i}: {b.text}" for i, b in enumerate(resume.bullets))
    prompt = f"Job: {job.title} at {job.company_name}\n\n{job.description}\n\nBullets:\n{bullets}"
    out = llm.structured(RequirementMatches, SYSTEM, prompt)
    rows: list[RequirementCoverage] = []
    for r in out.requirements:
        idx = sorted({i for i in r.bullet_indexes if 0 <= i < len(resume.bullets)})
        ev = (r.evidence or "").strip()
        ok = bool(idx) and bool(ev) and any(ev.lower() in resume.bullets[i].text.lower() for i in idx)
        rows.append(RequirementCoverage(
            requirement=r.requirement, covered=ok,
            bullet_indexes=idx if ok else [], evidence=ev if ok else None))
    gaps = [r.requirement for r in rows if not r.covered]
    ratio = (len(rows) - len(gaps)) / len(rows) if rows else 0.0
    return CoverageReport(requirements=rows, gaps=gaps, coverage_ratio=ratio)

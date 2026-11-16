"""Rewrite selected experience items into resume bullets."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel

from careerops.domain import ExperienceItem, Job, ResumeBullet, ResumeVersion
from careerops.llm import DEFAULT_MODEL, LLM

SYSTEM = (
    "You write resume bullets from a list of verified experience items. Rules: use only facts "
    "in the items. Do not add numbers, tools, team sizes, scope or seniority that are not in "
    "the items. Reword for the target job, do not invent. Each bullet lists the ids of the "
    "items it uses in source_item_ids."
)


class TailoredBullets(BaseModel):
    bullets: list[ResumeBullet]


def format_item(item: ExperienceItem) -> str:
    lines = [f"[{item.id}] {item.title}" + (f" at {item.employer}" if item.employer else "")]
    lines.append(f"  text: {item.text}")
    if item.metrics:
        lines.append(f"  metrics: {'; '.join(item.metrics)}")
    if item.skills:
        lines.append(f"  skills: {', '.join(item.skills)}")
    return "\n".join(lines)


def tailor_resume(job: Job, items: list[ExperienceItem], llm: LLM) -> ResumeVersion:
    """Ask the LLM for bullets and return an unverified ResumeVersion.

    Bullets that cite no ids, or ids outside `items`, are kept and noted so verify() rejects them.
    """
    prompt = (
        f"Target job: {job.title} at {job.company_name}\n\n{job.description}\n\n"
        "Experience items:\n" + "\n".join(format_item(i) for i in items)
    )
    out = llm.structured(TailoredBullets, SYSTEM, prompt)
    given = {i.id for i in items}
    notes = []
    for n, b in enumerate(out.bullets, 1):
        if not b.source_item_ids:
            notes.append(f"bullet {n}: cites no source items")
        elif bad := [s for s in b.source_item_ids if s not in given]:
            notes.append(f"bullet {n}: cites ids that were not provided: {bad}")
    return ResumeVersion(
        id=uuid.uuid4().hex[:12],
        job_id=job.id,
        created_at=datetime.now(UTC),
        bullets=out.bullets,
        verified=False,
        verification_notes=notes,
        model=DEFAULT_MODEL,
    )

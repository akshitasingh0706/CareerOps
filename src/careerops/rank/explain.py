"""LLM explanation and evidence mapping. The LLM writes text and picks experience item ids.
Code checks every id against the items it was given. The LLM produces no scores."""
from __future__ import annotations

from pydantic import BaseModel, Field

from careerops.domain import ExperienceItem, Job, UserProfile
from careerops.llm import LLM


class RequirementEvidence(BaseModel):
    requirement: str  # a requirement taken from the job posting
    item_ids: list[str] = Field(default_factory=list)  # ids of supporting experience items


class ExplanationOut(BaseModel):
    """Schema the LLM must fill."""
    explanation: str
    requirements: list[RequirementEvidence] = Field(default_factory=list)


class Explanation(BaseModel):
    """Validated result. Only ids that exist in the supplied items remain in `matches`."""
    explanation: str
    matches: list[RequirementEvidence]
    dropped_ids: list[str] = Field(default_factory=list)  # ids the LLM cited that do not exist
    error: str | None = None  # set when the LLM call failed and a fallback text was used

    def evidence_lines(self) -> list[str]:
        out = []
        for m in self.matches:
            ids = ", ".join(m.item_ids) if m.item_ids else "(no supporting experience)"
            out.append(f"{m.requirement} => {ids}")
        return out


SYSTEM = (
    "You assess how well a job fits a candidate. You are given the job, the candidate's wishes, "
    "numeric component scores already computed by code, and the candidate's verified experience "
    "items, each with an id. Write a short explanation (2 to 4 plain sentences) that cites facts "
    "from the job and the profile. Do not invent scores and do not restate numbers other than "
    "the ones given. Then list the main requirements of the job. For each, give the ids of the "
    "experience items that support it. Use only ids from the list. Use an empty list when no "
    "item supports the requirement."
)


def _prompt(
    job: Job, profile: UserProfile, items: list[ExperienceItem], components: dict[str, float]
) -> str:
    comp = "\n".join(f"- {k}: {v:.2f}" for k, v in components.items())
    exp = "\n".join(f"[{it.id}] {it.title}: {it.text} (skills: {', '.join(it.skills)})" for it in items)
    return (
        f"JOB: {job.title} at {job.company_name} ({job.location or 'location not stated'})\n"
        f"{job.description[:6000]}\n\n"
        f"CANDIDATE WISHES: {profile.model_dump_json(exclude_defaults=True)}\n\n"
        f"COMPONENT SCORES (0 to 1, computed by code):\n{comp}\n\n"
        f"EXPERIENCE ITEMS:\n{exp or '(none)'}"
    )


def validate_explanation(out: ExplanationOut, items: list[ExperienceItem]) -> Explanation:
    """Drop cited ids that are not in `items`. Keep each requirement even if all ids were dropped."""
    valid = {it.id for it in items}
    dropped: list[str] = []
    matches: list[RequirementEvidence] = []
    for req in out.requirements:
        kept: list[str] = []
        for i in req.item_ids:
            if i in valid:
                if i not in kept:
                    kept.append(i)
            elif i not in dropped:
                dropped.append(i)
        matches.append(RequirementEvidence(requirement=req.requirement, item_ids=kept))
    return Explanation(explanation=out.explanation.strip(), matches=matches, dropped_ids=dropped)


def explain_fit(
    llm: LLM,
    job: Job,
    profile: UserProfile,
    items: list[ExperienceItem],
    components: dict[str, float],
) -> Explanation:
    """Ask the LLM for an explanation and evidence, then validate the ids in code.
    If the call fails or returns empty text, return a plain fallback and set `error`."""
    try:
        out = llm.structured(ExplanationOut, SYSTEM, _prompt(job, profile, items, components))
        result = validate_explanation(out, items)
        if not result.explanation:
            raise ValueError("empty explanation")
        return result
    except Exception as exc:  # noqa: BLE001 - any failure falls back to code-written text
        best = max(components, key=components.get) if components else "none"
        return Explanation(
            explanation=f"No LLM explanation available. Strongest component: {best}.",
            matches=[],
            error=f"{type(exc).__name__}: {exc}",
        )

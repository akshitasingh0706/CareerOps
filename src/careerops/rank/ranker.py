"""Top-level ranking: filters, components, rules score, personalization, explanations."""
from __future__ import annotations

from pydantic import BaseModel, Field

from careerops.domain import ExperienceItem, FitScore, Job, UserProfile
from careerops.llm import LLM, Embedder

from .components import overall_score, score_components
from .config import RankConfig
from .explain import Explanation, explain_fit
from .feedback import FeedbackRow, FeedbackStore
from .filters import apply_filters, is_unknown, passes_all
from .personalize import PersonalizationInfo, Personalizer

SCORER_VERSION = "rank-v1"


class RankedJob(BaseModel):
    job: Job
    fit: FitScore
    rules_score: float
    learned_score: float | None = None
    final_score: float  # blended score used for sorting
    unknown_filters: list[str] = Field(default_factory=list)  # filters that need a manual check
    failed_filters: list[str] = Field(default_factory=list)
    explanation: Explanation | None = None


class RankResult(BaseModel):
    ranked: list[RankedJob]  # passed all filters, best first
    filtered_out: list[RankedJob]  # failed at least one filter, kept so the user can see why
    personalization: PersonalizationInfo


def build_fit(
    job: Job, components: dict[str, float], cfg: RankConfig, filters, explanation: Explanation | None
) -> FitScore:
    return FitScore(
        job_id=job.id,
        overall=overall_score(components, cfg.weights),
        components=components,
        filters=filters,
        explanation=explanation.explanation if explanation else "",
        evidence=explanation.evidence_lines() if explanation else [],
        scorer_version=SCORER_VERSION,
    )


def rank_jobs(
    jobs: list[Job],
    profile: UserProfile,
    items: list[ExperienceItem],
    llm: LLM,
    embedder: Embedder,
    *,
    feedback: FeedbackStore | list[FeedbackRow] | None = None,
    config: RankConfig | None = None,
) -> RankResult:
    """Rank jobs for a profile.

    Jobs that fail a hard filter go to `filtered_out` with their reasons. The rest are sorted
    by the blended score. Personalization is rules-only until there is enough feedback; the
    result says so in `personalization.status`. The LLM explains only the top `explain_top`
    ranked jobs. It never affects a score.
    """
    cfg = config or RankConfig()
    comps = score_components(jobs, profile, items, embedder, cfg.components)
    rows = feedback.rows() if isinstance(feedback, FeedbackStore) else (feedback or [])
    person = Personalizer.from_feedback(rows, cfg.personalization)

    ranked: list[RankedJob] = []
    out: list[RankedJob] = []
    for job, c in zip(jobs, comps, strict=True):
        filters = apply_filters(job, profile, cfg.filters)
        fit = build_fit(job, c, cfg, filters, None)
        final, learned = person.blend(fit.overall, c)
        rj = RankedJob(
            job=job,
            fit=fit,
            rules_score=fit.overall,
            learned_score=learned,
            final_score=final,
            unknown_filters=[f.name for f in filters if is_unknown(f)],
            failed_filters=[f.name for f in filters if not f.passed],
        )
        (ranked if passes_all(filters) else out).append(rj)

    ranked.sort(key=lambda r: (-r.final_score, r.job.id))
    out.sort(key=lambda r: (-r.final_score, r.job.id))

    limit = len(ranked) if cfg.explain_top is None else cfg.explain_top
    for rj in ranked[:limit]:
        exp = explain_fit(llm, rj.job, profile, items, rj.fit.components)
        rj.explanation = exp
        rj.fit = build_fit(rj.job, rj.fit.components, cfg, rj.fit.filters, exp)

    return RankResult(ranked=ranked, filtered_out=out, personalization=person.info)

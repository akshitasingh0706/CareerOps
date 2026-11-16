"""Fit components. Each is a float in 0..1 computed in code. No LLM is involved."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from careerops.domain import ExperienceItem, Job, UserProfile
from careerops.llm import Embedder

from .config import ComponentConfig, ScoringWeights
from .filters import SENIORITY_ORDER, _is_remote, _location_match, offered_salary, seniority_range
from .text import contains_term, tokens

_MAX_SENIORITY_DISTANCE = 3.0


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def job_text(job: Job) -> str:
    return f"{job.title}\n{job.description}"


def profile_text(profile: UserProfile, items: list[ExperienceItem]) -> str:
    parts = list(profile.target_titles)
    for it in items:
        parts.append(f"{it.title}. {it.text} {' '.join(it.skills)}")
    return "\n".join(parts)


def user_skills(profile: UserProfile, items: list[ExperienceItem]) -> list[str]:
    seen: dict[str, str] = {}
    for s in [*profile.must_have_skills, *(s for it in items for s in it.skills)]:
        if s.strip():
            seen.setdefault(s.strip().lower(), s.strip())
    return list(seen.values())


@dataclass
class ProfileContext:
    """Profile-side data prepared once and reused for every job."""
    profile: UserProfile
    skills: list[str]
    profile_vec: np.ndarray | None
    interest_vecs: np.ndarray | None


def build_context(
    profile: UserProfile, items: list[ExperienceItem], embedder: Embedder
) -> ProfileContext:
    ptext = profile_text(profile, items)
    texts = ([ptext] if ptext.strip() else []) + list(profile.interests)
    vecs = embedder.embed(texts) if texts else np.zeros((0, embedder.dim))
    has_p = bool(ptext.strip())
    return ProfileContext(
        profile=profile,
        skills=user_skills(profile, items),
        profile_vec=vecs[0] if has_p else None,
        interest_vecs=vecs[1 if has_p else 0:] if profile.interests else None,
    )


def _sim(a: np.ndarray, b: np.ndarray, cfg: ComponentConfig) -> float:
    """Cosine similarity rescaled so sim_floor gives 0 and sim_ceiling gives 1."""
    cos = float(np.dot(a, b))
    return _clip((cos - cfg.sim_floor) / (cfg.sim_ceiling - cfg.sim_floor))


def skills_component(
    job: Job, ctx: ProfileContext, job_vec: np.ndarray, cfg: ComponentConfig
) -> float:
    """share * embedding + (1 - share) * keywords.
    Embedding: rescaled cosine between the job text and the profile text (target titles plus
    experience items). Keywords: matched user skills in the job text divided by
    min(number of user skills, skill_hits_cap). With no user skills, the embedding alone is used."""
    emb = _sim(job_vec, ctx.profile_vec, cfg) if ctx.profile_vec is not None else cfg.neutral
    if not ctx.skills:
        return emb
    text = job_text(job)
    hits = sum(1 for s in ctx.skills if contains_term(text, s))
    kw = _clip(hits / min(len(ctx.skills), cfg.skill_hits_cap))
    return _clip(cfg.embedding_share * emb + (1 - cfg.embedding_share) * kw)


def seniority_component(job: Job, profile: UserProfile, cfg: ComponentConfig) -> float:
    """1 - d/3, where d is the ladder distance from the job level to the middle of the user's
    range. Ladder: intern 0, junior 1, mid 2, senior and manager 3, staff 4, principal 5.
    Neutral when the job level or the user's range is unknown."""
    rng = seniority_range(profile)
    pos = SENIORITY_ORDER.get(job.seniority)
    if rng is None or pos is None:
        return cfg.neutral
    mid = (rng[0] + rng[1]) / 2
    return _clip(1 - abs(pos - mid) / _MAX_SENIORITY_DISTANCE)

def compensation_component(job: Job, profile: UserProfile, cfg: ComponentConfig) -> float:
    """clip((r - 0.6) / 0.7, 0, 1) where r = top of posted range / user minimum.
    So r=0.6 gives 0, r=1 gives about 0.57, r=1.3 or more gives 1.
    Neutral when there is no minimum, no posted salary, or the currencies differ."""
    offer = offered_salary(job)
    if profile.min_salary is None or offer is None:
        return cfg.neutral
    if job.salary_currency and job.salary_currency != profile.salary_currency:
        return cfg.neutral
    return _clip((offer / profile.min_salary - 0.6) / 0.7)


def location_component(job: Job, profile: UserProfile, cfg: ComponentConfig) -> float:
    """1.0 for a remote job when remote is ok, or when the job location matches one of the
    user's locations. 0.0 for a known location that does not match. Neutral when the job
    gives no location and no remote flag, or the user has no location preference."""
    remote = _is_remote(job)
    if remote and profile.remote_ok:
        return 1.0
    if _location_match(job, profile):
        return 1.0
    if remote is None and not job.location:
        return cfg.neutral
    if not profile.locations and profile.remote_ok:
        return cfg.neutral
    return 0.0


def growth_component(
    job: Job, ctx: ProfileContext, job_vec: np.ndarray, cfg: ComponentConfig
) -> float:
    """Match between the job and the user's interests, using the best-matching interest.
    share * max rescaled cosine + (1 - share) * max keyword coverage, where coverage is the
    fraction of an interest's words found in the job text. Neutral if there are no interests."""
    interests = ctx.profile.interests
    if not interests or ctx.interest_vecs is None:
        return cfg.neutral
    emb = max(_sim(job_vec, v, cfg) for v in ctx.interest_vecs)
    jt = tokens(job_text(job))
    kw = 0.0
    for text in interests:
        toks = tokens(text)
        if toks:
            kw = max(kw, len(toks & jt) / len(toks))
    return _clip(cfg.embedding_share * emb + (1 - cfg.embedding_share) * kw)


def compute_components(
    job: Job, ctx: ProfileContext, job_vec: np.ndarray, cfg: ComponentConfig | None = None
) -> dict[str, float]:
    cfg = cfg or ComponentConfig()
    return {
        "skills": skills_component(job, ctx, job_vec, cfg),
        "seniority": seniority_component(job, ctx.profile, cfg),
        "compensation": compensation_component(job, ctx.profile, cfg),
        "location": location_component(job, ctx.profile, cfg),
        "growth": growth_component(job, ctx, job_vec, cfg),
    }


def overall_score(components: dict[str, float], weights: ScoringWeights | None = None) -> float:
    """Weighted sum of components divided by the sum of weights. Result is in 0..1."""
    w = weights or ScoringWeights()
    total = sum(getattr(w, k) for k in components)
    return _clip(sum(getattr(w, k) * v for k, v in components.items()) / total)


def score_components(
    jobs: list[Job],
    profile: UserProfile,
    items: list[ExperienceItem],
    embedder: Embedder,
    cfg: ComponentConfig | None = None,
) -> list[dict[str, float]]:
    """Components for many jobs. Embeds all job texts in one batch."""
    ctx = build_context(profile, items, embedder)
    if not jobs:
        return []
    vecs = embedder.embed([job_text(j) for j in jobs])
    return [compute_components(j, ctx, vecs[i], cfg) for i, j in enumerate(jobs)]

"""Plain config objects for ranking. Change values here, not in the scoring code."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

COMPONENT_NAMES = ("skills", "seniority", "compensation", "location", "growth")

# What to do when a job lacks the data a filter needs.
#   "unknown": filter passes, detail starts with "unknown:", job is listed as needing a check
#   "pass":    filter passes quietly
#   "fail":    filter fails
MissingPolicy = Literal["unknown", "pass", "fail"]


class FilterConfig(BaseModel):
    compensation_missing: MissingPolicy = "unknown"
    location_missing: MissingPolicy = "unknown"
    seniority_missing: MissingPolicy = "unknown"


class ScoringWeights(BaseModel):
    """Weights for the rules score. They are divided by their sum, so they need not add to 1."""
    skills: float = 0.35
    seniority: float = 0.15
    compensation: float = 0.15
    location: float = 0.15
    growth: float = 0.20

    @model_validator(mode="after")
    def _check(self) -> ScoringWeights:
        vals = [getattr(self, n) for n in COMPONENT_NAMES]
        if any(v < 0 for v in vals) or sum(vals) <= 0:
            raise ValueError("weights must be non-negative and not all zero")
        return self


class ComponentConfig(BaseModel):
    # Cosine similarity is rescaled so floor maps to 0 and ceiling maps to 1.
    sim_floor: float = 0.0
    sim_ceiling: float = 0.5
    embedding_share: float = 0.5  # share of embedding in skills and growth; the rest is keywords
    skill_hits_cap: int = 8  # this many matched skills gives a full keyword score
    neutral: float = 0.5  # score when the job or the profile gives no information


class PersonalizationConfig(BaseModel):
    min_feedback: int = 10  # total feedback rows needed before a model is fit
    min_per_class: int = 2  # positive rows and negative rows both needed
    l2: float = 1.0  # ridge strength on the weights (not the intercept)
    max_blend: float = 0.8  # the learned score never takes more than this share
    half_blend_at: int = 30  # feedback count at which the learned share is half of max_blend


class RankConfig(BaseModel):
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    components: ComponentConfig = Field(default_factory=ComponentConfig)
    personalization: PersonalizationConfig = Field(default_factory=PersonalizationConfig)
    explain_top: int | None = 10  # LLM explanations for the top N passing jobs; None means all

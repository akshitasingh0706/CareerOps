"""Learned ranking from feedback. A small L2-regularized logistic regression in numpy over the
component vectors, blended with the rules score."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from .config import COMPONENT_NAMES, PersonalizationConfig
from .feedback import FeedbackRow


class PersonalizationInfo(BaseModel):
    """What the user sees about personalization."""
    status: str  # "insufficient_data", "rules_only" or "active"
    n_feedback: int = 0
    n_positive: int = 0
    n_negative: int = 0
    needed: str = ""  # plain reason when status is not "active"
    learned_weight: float = 0.0  # share of the learned score in the blended score, 0..1
    coefficients: dict[str, float] = Field(default_factory=dict)  # per component, on 0..1 inputs
    intercept: float = 0.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, iters: int = 100) -> np.ndarray:
    """Newton's method for logistic regression with a ridge penalty on all but the intercept.
    x is (n, d) without an intercept column. Returns a vector of length d + 1, intercept last."""
    n, d = x.shape
    xb = np.hstack([x, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.eye(d + 1) * l2
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = _sigmoid(xb @ w)
        grad = xb.T @ (p - y) + reg @ w
        hess = (xb.T * (p * (1 - p))) @ xb + reg + np.eye(d + 1) * 1e-9
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def _vector(components: dict[str, float]) -> np.ndarray:
    return np.array([components[n] for n in COMPONENT_NAMES], dtype=float)


class Personalizer:
    """A fitted model plus the blend rule. Build it with `Personalizer.from_feedback`."""

    def __init__(self, w: np.ndarray | None, info: PersonalizationInfo, cfg: PersonalizationConfig):
        self._w = w
        self.info = info
        self.cfg = cfg

    @classmethod
    def from_feedback(
        cls, rows: list[FeedbackRow], cfg: PersonalizationConfig | None = None
    ) -> Personalizer:
        cfg = cfg or PersonalizationConfig()
        y = np.array([r.label for r in rows], dtype=float)
        pos, neg = int(y.sum()), int(len(y) - y.sum())
        base = {"n_feedback": len(rows), "n_positive": pos, "n_negative": neg}
        if len(rows) < cfg.min_feedback or pos < cfg.min_per_class or neg < cfg.min_per_class:
            need = (
                f"need at least {cfg.min_feedback} feedback rows with {cfg.min_per_class} "
                f"positive and {cfg.min_per_class} negative; have {len(rows)} "
                f"({pos} positive, {neg} negative)"
            )
            return cls(None, PersonalizationInfo(status="insufficient_data", needed=need, **base), cfg)
        x = np.vstack([_vector(r.components) for r in rows])
        w = fit_logistic(x, y, cfg.l2)
        alpha = cfg.max_blend * len(rows) / (len(rows) + cfg.half_blend_at)
        info = PersonalizationInfo(
            status="active",
            learned_weight=float(alpha),
            coefficients={n: float(w[i]) for i, n in enumerate(COMPONENT_NAMES)},
            intercept=float(w[-1]),
            **base,
        )
        return cls(w, info, cfg)

    @property
    def active(self) -> bool:
        return self._w is not None

    def learned_score(self, components: dict[str, float]) -> float | None:
        """Predicted probability of a positive action, or None when not active."""
        if self._w is None:
            return None
        z = float(_vector(components) @ self._w[:-1] + self._w[-1])
        return float(_sigmoid(np.array(z)))

    def blend(self, rules: float, components: dict[str, float]) -> tuple[float, float | None]:
        """Return (blended score, learned score). Blended = (1 - a) * rules + a * learned."""
        learned = self.learned_score(components)
        if learned is None:
            return rules, None
        a = self.info.learned_weight
        return (1 - a) * rules + a * learned, learned

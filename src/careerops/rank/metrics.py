"""Ranking metrics."""
from __future__ import annotations

import math


def dcg(rels: list[float]) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rels: list[float], k: int = 10) -> float:
    """NDCG@k. `ranked_rels` holds the true relevance of each item in the order the ranker
    put them. The ideal order is the same list sorted high to low."""
    ideal = dcg(sorted(ranked_rels, reverse=True)[:k])
    return dcg(ranked_rels[:k]) / ideal if ideal > 0 else 0.0

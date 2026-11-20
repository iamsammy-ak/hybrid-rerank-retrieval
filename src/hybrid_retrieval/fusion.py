"""Combine the sparse and dense ranked lists into one.

* **Reciprocal Rank Fusion (RRF)**: ``score(d) = sum_i w_i / (k + rank_i(d))`` with 1-based ranks. It uses
  only positions, so the two retrievers' incompatible score scales never meet. A larger ``k`` flattens the
  advantage of the very top ranks.
* **Convex combination**: ``score(d) = alpha * dense'(d) + (1 - alpha) * sparse'(d)`` where ``'`` is a per-query
  normalisation of each list (min-max to [0, 1], or z-score). A document missing from one list was not retrieved
  by it, so it gets the score of that list's worst hit: 0 under min-max, the lowest z-score under z-score (-1
  when the list is constant, since every hit then has z = 0).

Ties in the fused score are broken by document position so the output is deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from .schemas import Ranked

Normalisation = Literal["minmax", "zscore"]


def normalise(scores: np.ndarray, method: Normalisation = "minmax") -> np.ndarray:
    """Per-list score normalisation. A constant list maps to 1.0 (min-max) or 0.0 (z-score)."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    if method == "minmax":
        lo, hi = float(scores.min()), float(scores.max())
        return np.ones_like(scores) if hi == lo else (scores - lo) / (hi - lo)
    if method == "zscore":
        std = float(scores.std())
        return np.zeros_like(scores) if std == 0.0 else (scores - scores.mean()) / std
    raise ValueError(f"unknown normalisation {method!r}")


def _sorted(positions: np.ndarray, scores: np.ndarray) -> Ranked:
    order = np.lexsort((positions, -scores))
    return Ranked(positions[order].astype(np.int64), scores[order].astype(np.float64))


def rrf(rankings: Sequence[Ranked], k: float = 60.0, weights: Sequence[float] | None = None) -> Ranked:
    """Reciprocal Rank Fusion of any number of ranked lists."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    weights = [1.0] * len(rankings) if weights is None else list(weights)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking is required")
    parts = [r for r in rankings if len(r)]
    if not parts:
        return Ranked.empty()
    positions = np.concatenate([r.positions for r in parts])
    contrib = np.concatenate([w / (k + np.arange(1, len(r) + 1, dtype=np.float64))
                              for r, w in zip(rankings, weights, strict=True) if len(r)])
    unique, inverse = np.unique(positions, return_inverse=True)
    fused = np.bincount(inverse, weights=contrib, minlength=unique.size)
    return _sorted(unique, fused)


def convex(sparse: Ranked, dense: Ranked, alpha: float = 0.5, norm: Normalisation = "minmax") -> Ranked:
    """Convex combination of normalised scores; ``alpha`` is the weight on the dense list."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    union = np.union1d(sparse.positions, dense.positions)
    if union.size == 0:
        return Ranked.empty()

    def aligned(ranked: Ranked) -> np.ndarray:
        values = normalise(ranked.scores, norm)
        if norm == "minmax" or values.size == 0:
            floor = 0.0
        else:
            floor = float(values.min()) if float(values.std()) > 0.0 else -1.0
        out = np.full(union.size, floor)
        out[np.searchsorted(union, ranked.positions)] = values
        return out

    fused = alpha * aligned(dense) + (1.0 - alpha) * aligned(sparse)
    return _sorted(union, fused)

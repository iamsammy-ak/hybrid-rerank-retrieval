"""Top-k selection shared by every retriever."""

from __future__ import annotations

import numpy as np

from .schemas import Ranked


def top_k(scores: np.ndarray, k: int, allowed: np.ndarray | None = None, min_score: float | None = None) -> Ranked:
    """Return the ``k`` highest-scoring positions, best first.

    ``allowed`` is an optional boolean mask over positions: only those can be returned (this is how pre-filtering
    works). ``min_score`` drops positions whose score is not strictly greater. Ties are broken by the lower
    position, so results are deterministic. Runs in O(n) with ``argpartition`` plus O(k log k) for the final sort.
    """
    if allowed is not None:
        idx: np.ndarray | None = np.flatnonzero(allowed)
        values = scores[idx]
    else:
        idx, values = None, scores
    if min_score is not None:
        keep = values > min_score
        idx = np.flatnonzero(keep) if idx is None else idx[keep]
        values = values[keep]
    n = int(values.shape[0])
    if k <= 0 or n == 0:
        return Ranked.empty()
    if k < n:
        kth = values[np.argpartition(-values, k - 1)[k - 1]]
        above = np.flatnonzero(values > kth)
        ties = np.flatnonzero(values == kth)[: k - above.size]
        selected = np.concatenate([above, ties])
    else:
        selected = np.arange(n)
    selected = selected[np.lexsort((selected, -values[selected]))]
    positions = selected if idx is None else idx[selected]
    return Ranked(positions.astype(np.int64), values[selected].astype(np.float64))

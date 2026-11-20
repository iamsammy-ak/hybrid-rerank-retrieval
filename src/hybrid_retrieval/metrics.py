"""Ranking-quality metrics (trec_eval conventions, as used by BEIR) and latency summaries.

* **nDCG@k** (normalised discounted cumulative gain): ``DCG = sum(rel_i / log2(i + 1))`` over the first ``k``
  results, divided by the DCG of the ideal ordering. Gains are the graded relevance labels (linear gain, as in
  trec_eval's ``ndcg_cut``).
* **Recall@k**: share of the judged-relevant documents that appear in the first ``k`` results.
* **MRR@k** (mean reciprocal rank): ``1 / rank`` of the first relevant result within ``k``, else 0.

Every metric is averaged over all judged queries; a query with no results scores 0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Sequence[str]]


def ndcg_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int = 10) -> float:
    """nDCG@k of one ranked list against graded judgements (grades <= 0 count as non-relevant)."""
    gains = {doc: g for doc, g in relevant.items() if g > 0}
    if not gains:
        return 0.0
    dcg = sum(gains.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(ranking[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg


def recall_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int = 100) -> float:
    """Recall@k of one ranked list."""
    rel = {doc for doc, g in relevant.items() if g > 0}
    if not rel:
        return 0.0
    return len(rel.intersection(ranking[:k])) / len(rel)


def mrr_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int = 10) -> float:
    """Reciprocal rank of the first relevant document within the first ``k`` results."""
    for i, doc in enumerate(ranking[:k]):
        if relevant.get(doc, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(run: Run, qrels: Qrels) -> dict[str, float]:
    """Mean nDCG@10, Recall@100 and MRR@10 over every query in ``qrels``."""
    if not qrels:
        raise ValueError("qrels is empty")
    totals = {"ndcg@10": 0.0, "recall@100": 0.0, "mrr@10": 0.0}
    for qid, relevant in qrels.items():
        ranking = run.get(qid, ())
        totals["ndcg@10"] += ndcg_at_k(ranking, relevant, 10)
        totals["recall@100"] += recall_at_k(ranking, relevant, 100)
        totals["mrr@10"] += mrr_at_k(ranking, relevant, 10)
    return {name: value / len(qrels) for name, value in totals.items()}


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    """p50, p95 and mean of a latency sample, in milliseconds."""
    if len(samples_ms) == 0:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)), "mean": float(arr.mean())}

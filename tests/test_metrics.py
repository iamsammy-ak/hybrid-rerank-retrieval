from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hybrid_retrieval.metrics import evaluate, latency_summary, mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_perfect_and_worst() -> None:
    rel = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], rel) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], rel) == 0.0
    assert ndcg_at_k([], rel) == 0.0
    assert ndcg_at_k(["a"], {}) == 0.0


def test_ndcg_hand_computed_graded() -> None:
    rel = {"a": 2, "b": 1}
    dcg = 1 / math.log2(2) + 0 + 2 / math.log2(4)  # ranking b, x, a
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["b", "x", "a"], rel) == pytest.approx(dcg / idcg)


def test_ndcg_cutoff() -> None:
    assert ndcg_at_k(["x", "a"], {"a": 1}, k=1) == 0.0


def test_recall_and_mrr() -> None:
    rel = {"a": 1, "b": 1, "c": 0}
    assert recall_at_k(["a", "x", "c"], rel, k=100) == 0.5
    assert recall_at_k(["a", "b"], rel, k=1) == 0.5
    assert mrr_at_k(["x", "y", "b"], rel) == pytest.approx(1 / 3)
    assert mrr_at_k(["x"] * 10 + ["a"], rel, k=10) == 0.0
    assert recall_at_k(["a"], {}) == 0.0


def test_evaluate_averages_and_counts_missing_queries_as_zero() -> None:
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    scores = evaluate({"q1": ["a"]}, qrels)
    assert scores["ndcg@10"] == pytest.approx(0.5)
    assert scores["recall@100"] == pytest.approx(0.5)
    assert scores["mrr@10"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        evaluate({}, {})


docs = st.lists(st.sampled_from("abcdefghij"), unique=True, max_size=10)


@given(docs, st.dictionaries(st.sampled_from("abcdefghij"), st.integers(0, 3), max_size=6))
def test_metrics_are_bounded(ranking: list[str], rel: dict[str, int]) -> None:
    for value in (ndcg_at_k(ranking, rel), recall_at_k(ranking, rel), mrr_at_k(ranking, rel)):
        assert 0.0 <= value <= 1.0 + 1e-12


@given(st.dictionaries(st.sampled_from("abcdefghij"), st.integers(1, 3), min_size=1, max_size=6))
def test_ideal_ordering_scores_one(rel: dict[str, int]) -> None:
    ideal = sorted(rel, key=rel.__getitem__, reverse=True)
    assert ndcg_at_k(ideal, rel) == pytest.approx(1.0)


def test_latency_summary() -> None:
    out = latency_summary([1.0, 2.0, 3.0, 4.0, 100.0])
    assert out["p50"] == 3.0
    assert out["p95"] == pytest.approx(80.8)
    assert out["mean"] == pytest.approx(22.0)
    assert latency_summary([]) == {"p50": 0.0, "p95": 0.0, "mean": 0.0}

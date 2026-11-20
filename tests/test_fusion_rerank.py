from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hybrid_retrieval.fusion import convex, normalise, rrf
from hybrid_retrieval.rerank import TermOverlapReranker, load_reranker, rerank_head
from hybrid_retrieval.schemas import Ranked


def ranked(positions: list[int], scores: list[float] | None = None) -> Ranked:
    scores = scores if scores is not None else [float(len(positions) - i) for i in range(len(positions))]
    return Ranked(np.array(positions, dtype=np.int64), np.array(scores, dtype=np.float64))


def test_rrf_hand_computed() -> None:
    fused = rrf([ranked([1, 2, 3]), ranked([3, 1, 4])], k=60)
    expected = {1: 1 / 61 + 1 / 62, 2: 1 / 62, 3: 1 / 63 + 1 / 61, 4: 1 / 63}
    assert fused.positions.tolist() == [1, 3, 2, 4]
    for pos, score in zip(fused.positions, fused.scores, strict=False):
        assert score == pytest.approx(expected[int(pos)])


def test_rrf_small_k_rewards_top_ranks() -> None:
    a, b = ranked([10, 20, 30]), ranked([30, 40, 20])
    assert rrf([a, b], k=1).positions[0] == 30
    assert rrf([a, b], k=1, weights=[3.0, 1.0]).positions[0] == 10


def test_rrf_validation_and_empty() -> None:
    with pytest.raises(ValueError):
        rrf([ranked([1])], k=0)
    with pytest.raises(ValueError):
        rrf([ranked([1]), ranked([2])], weights=[1.0])
    assert len(rrf([Ranked.empty(), Ranked.empty()])) == 0
    assert rrf([Ranked.empty(), ranked([5, 6])]).positions.tolist() == [5, 6]


lists = st.lists(st.integers(0, 30), max_size=15, unique=True)


@given(lists, lists, st.floats(0.5, 200))
def test_rrf_properties(a: list[int], b: list[int], k: float) -> None:
    fused = rrf([ranked(a), ranked(b)], k=k)
    assert set(fused.positions.tolist()) == set(a) | set(b)
    assert (np.diff(fused.scores) <= 1e-15).all()


def test_normalise() -> None:
    np.testing.assert_allclose(normalise(np.array([2.0, 4.0, 6.0])), [0.0, 0.5, 1.0])
    z = normalise(np.array([1.0, 2.0, 3.0, 10.0]), "zscore")
    assert z.mean() == pytest.approx(0.0) and z.std() == pytest.approx(1.0)
    np.testing.assert_array_equal(normalise(np.array([3.0, 3.0])), [1.0, 1.0])
    np.testing.assert_array_equal(normalise(np.array([3.0, 3.0]), "zscore"), [0.0, 0.0])
    assert normalise(np.array([])).size == 0
    with pytest.raises(ValueError):
        normalise(np.array([1.0]), "softmax")  # type: ignore[arg-type]


def test_convex_extremes_follow_one_list() -> None:
    sparse = ranked([1, 2, 3], [9.0, 5.0, 1.0])
    dense = ranked([3, 2, 4], [0.9, 0.8, 0.1])
    assert convex(sparse, dense, alpha=0.0).positions[:3].tolist() == [1, 2, 3]
    dense_only = convex(sparse, dense, alpha=1.0)
    assert dense_only.positions[:2].tolist() == [3, 2]
    scores = dict(zip(dense_only.positions.tolist(), dense_only.scores.tolist(), strict=False))
    assert scores[4] == scores[1] == 0.0  # the worst dense hit ties with documents the dense list never returned


def test_convex_missing_docs_get_list_floor() -> None:
    sparse = ranked([1, 2], [10.0, 0.0])
    dense = ranked([3], [0.7])
    fused = convex(sparse, dense, alpha=0.5)
    scores = dict(zip(fused.positions.tolist(), fused.scores.tolist(), strict=False))
    # min-max: doc 1 = 0.5*1 (sparse) + 0.5*0 (missing from dense); doc 3 = 0.5*0 (missing) + 0.5*1; doc 2 = 0
    assert scores[1] == pytest.approx(0.5)
    assert scores[3] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(0.0)
    fused_z = convex(sparse, dense, 0.5, "zscore")
    z = dict(zip(fused_z.positions.tolist(), fused_z.scores.tolist(), strict=False))
    assert z[3] == pytest.approx(0.5 * 0.0 + 0.5 * -1.0)  # dense hit z = 0; missing from sparse -> lowest z = -1
    assert z[1] == pytest.approx(0.5 * 1.0 + 0.5 * -1.0)  # constant dense list -> missing docs get -1


def test_convex_validation_and_empty() -> None:
    with pytest.raises(ValueError):
        convex(ranked([1]), ranked([2]), alpha=1.5)
    assert len(convex(Ranked.empty(), Ranked.empty())) == 0


@given(lists, lists, st.floats(0, 1), st.sampled_from(["minmax", "zscore"]))
def test_convex_properties(a: list[int], b: list[int], alpha: float, norm: str) -> None:
    fused = convex(ranked(a), ranked(b), alpha=alpha, norm=norm)  # type: ignore[arg-type]
    assert set(fused.positions.tolist()) == set(a) | set(b)
    assert (np.diff(fused.scores) <= 1e-12).all()


def test_rerank_head_reorders_only_the_head() -> None:
    base = ranked([10, 11, 12, 13, 14])
    out = rerank_head(base, 3, np.array([0.1, 0.9, 0.5]))
    assert out.positions.tolist() == [11, 12, 10, 13, 14]
    assert (np.diff(out.scores) < 0).all()
    with pytest.raises(ValueError):
        rerank_head(base, 3, np.array([1.0]))


def test_rerank_head_depth_larger_than_list() -> None:
    out = rerank_head(ranked([1, 2]), 10, np.array([0.0, 1.0]))
    assert out.positions.tolist() == [2, 1]


def test_term_overlap_reranker() -> None:
    rr = TermOverlapReranker()
    scores = rr.score("rotate leaked keys", ["How to rotate leaked API keys", "Rotate the screen", "nothing"])
    assert scores[0] > scores[1] > scores[2] == 0.0
    assert rr.score("the of", ["anything"]).tolist() == [0.0]
    assert isinstance(load_reranker("overlap"), TermOverlapReranker)
    with pytest.raises(ValueError):
        load_reranker("gpt")

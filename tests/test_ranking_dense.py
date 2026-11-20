from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hybrid_retrieval.dense import DenseIndex
from hybrid_retrieval.embedders import HashingEmbedder, load_embedder
from hybrid_retrieval.ranking import top_k
from hybrid_retrieval.schemas import Ranked

scores_strategy = arrays(np.float64, st.integers(1, 60), elements=st.integers(-5, 5).map(float))


@settings(max_examples=150)
@given(scores_strategy, st.integers(0, 70))
def test_top_k_matches_stable_full_sort(scores: np.ndarray, k: int) -> None:
    expected = np.lexsort((np.arange(scores.size), -scores))[:k]
    ranked = top_k(scores, k)
    assert ranked.positions.tolist() == expected.tolist()
    np.testing.assert_array_equal(ranked.scores, scores[expected])


@given(scores_strategy, st.integers(1, 10), st.data())
def test_top_k_respects_mask_and_min_score(scores: np.ndarray, k: int, data: st.DataObject) -> None:
    mask = data.draw(arrays(np.bool_, scores.size))
    ranked = top_k(scores, k, allowed=mask, min_score=0.0)
    assert all(mask[p] and scores[p] > 0 for p in ranked.positions)
    assert len(ranked) == min(k, int((mask & (scores > 0)).sum()))


def test_ranked_validation_and_head() -> None:
    with pytest.raises(ValueError):
        Ranked(np.arange(3), np.zeros(2))
    r = Ranked(np.arange(5), np.linspace(1, 0, 5))
    assert r.head(2).positions.tolist() == [0, 1]
    assert len(Ranked.empty()) == 0


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder(dim=64)
    a = emb.embed_documents(["Cross-encoder reranking", "BM25 sparse retrieval", ""])
    b = HashingEmbedder(dim=64).embed_documents(["Cross-encoder reranking", "BM25 sparse retrieval", ""])
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float32 and a.shape == (3, 64)
    np.testing.assert_allclose(np.linalg.norm(a[:2], axis=1), 1.0, rtol=1e-5)
    assert np.linalg.norm(a[2]) == 0.0  # empty text has no features
    assert emb.embed_documents([]).shape == (0, 64)


def test_hashing_embedder_prefers_shared_words() -> None:
    emb = HashingEmbedder()
    q = emb.embed_query("reranking with cross encoders")
    docs = emb.embed_documents(["a cross-encoder reranks candidates", "the weather in Turin is mild"])
    assert docs[0] @ q > docs[1] @ q


def test_embedder_validation() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dim=4)
    with pytest.raises(ValueError):
        HashingEmbedder(ngram_range=(4, 2))
    with pytest.raises(ValueError):
        load_embedder("word2vec")


def _random_unit(n: int, d: int, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_dense_search_is_exact() -> None:
    vectors = _random_unit(300, 32)
    index = DenseIndex(vectors)
    q = vectors[17]
    ranked = index.search(q, k=5)
    assert ranked.positions[0] == 17
    assert ranked.positions.tolist() == np.argsort(-(vectors @ q), kind="stable")[:5].tolist()


def test_dense_pre_filter() -> None:
    vectors = _random_unit(50, 16)
    allowed = np.zeros(50, dtype=bool)
    allowed[[3, 9, 20]] = True
    ranked = DenseIndex(vectors).search(vectors[0], k=10, allowed=allowed)
    assert sorted(ranked.positions.tolist()) == [3, 9, 20]


def test_int8_is_four_times_smaller_and_close() -> None:
    vectors = _random_unit(2000, 64, seed=1)
    exact = DenseIndex(vectors)
    quant = exact.quantise_int8()
    assert quant.quantised and quant.vectors.dtype == np.int8
    assert exact.nbytes / quant.nbytes > 3.9
    q = vectors[5]
    assert np.abs(exact.score(q) - quant.score(q)).max() < 0.02
    overlap = set(exact.search(q, 10).positions) & set(quant.search(q, 10).positions)
    assert len(overlap) >= 8
    assert quant.quantise_int8() is quant


def test_dense_save_load(tmp_path: Path) -> None:
    exact = DenseIndex(_random_unit(20, 8))
    exact.save(tmp_path)
    exact.quantise_int8().save(tmp_path)
    np.testing.assert_array_equal(DenseIndex.load(tmp_path).vectors, exact.vectors)
    loaded_q = DenseIndex.load(tmp_path, quantised=True)
    assert loaded_q.quantised and loaded_q.n_docs == 20


def test_dense_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        DenseIndex(np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError):
        DenseIndex(np.zeros((3, 4), dtype=np.float32), scale=np.ones(4))
    with pytest.raises(ValueError):
        DenseIndex(_random_unit(3, 4)).score(np.zeros(5))

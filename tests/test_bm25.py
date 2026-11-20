from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hybrid_retrieval.bm25 import BM25Index, BM25Params
from hybrid_retrieval.tokenizer import Tokenizer, TokenizerConfig

PLAIN = TokenizerConfig(remove_stopwords=False, stemmer="none")


def naive_bm25(texts: list[str], query: str, k1: float, b: float) -> np.ndarray:
    """Textbook BM25 with explicit loops, used as the reference implementation."""
    tok = Tokenizer(PLAIN)
    docs = [Counter(tok(t)) for t in texts]
    lengths = [sum(d.values()) for d in docs]
    avgdl = sum(lengths) / len(docs)
    n = len(docs)
    out = np.zeros(n)
    for term, qtf in Counter(tok(query)).items():
        df = sum(1 for d in docs if term in d)
        if df == 0:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, d in enumerate(docs):
            tf = d.get(term, 0)
            if tf:
                out[i] += qtf * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[i] / avgdl))
    return out


CORPUS = [
    "the cat sat on the mat",
    "the dog chased the cat around the garden",
    "a quick brown fox jumps over the lazy dog",
    "cats and dogs are common pets pets pets",
]


@pytest.mark.parametrize(("k1", "b"), [(0.9, 0.4), (1.2, 0.75), (0.0, 0.5), (2.0, 0.0), (1.5, 1.0)])
def test_matches_reference(k1: float, b: float) -> None:
    index = BM25Index.build(CORPUS, PLAIN, BM25Params(k1=k1, b=b))
    for query in ["cat", "the dog", "pets pets", "fox garden mat", "unknown words"]:
        np.testing.assert_allclose(index.score(query), naive_bm25(CORPUS, query, k1, b), rtol=1e-5, atol=1e-6)


words = st.sampled_from(["alpha", "beta", "gamma", "delta", "eps", "zeta"])
docs_strategy = st.lists(st.lists(words, min_size=1, max_size=12).map(" ".join), min_size=1, max_size=8)


@settings(max_examples=60, deadline=None)
@given(docs_strategy, st.lists(words, min_size=1, max_size=4).map(" ".join),
       st.floats(0.1, 3.0), st.floats(0.0, 1.0))
def test_vectorised_equals_reference(texts: list[str], query: str, k1: float, b: float) -> None:
    index = BM25Index.build(texts, PLAIN, BM25Params(k1=k1, b=b))
    np.testing.assert_allclose(index.score(query), naive_bm25(texts, query, k1, b), rtol=1e-4, atol=1e-5)


def test_idf_is_positive_even_for_common_terms() -> None:
    index = BM25Index.build(["the cat", "the dog", "the end"], PLAIN)
    assert (index.idf > 0).all()


def test_search_ranks_and_drops_zero_scores() -> None:
    index = BM25Index.build(CORPUS, PLAIN)
    ranked = index.search("cat mat", k=10)
    assert ranked.positions[0] == 0
    assert set(ranked.positions.tolist()) == {0, 1}
    assert (np.diff(ranked.scores) <= 0).all()


def test_empty_and_unknown_queries() -> None:
    index = BM25Index.build(CORPUS)
    assert len(index.search("", k=5)) == 0
    assert len(index.search("zzzz qqqq", k=5)) == 0
    assert index.query_vector("the of and")[0].size == 0  # only stopwords


def test_allowed_mask_restricts_results() -> None:
    index = BM25Index.build(CORPUS, PLAIN)
    allowed = np.array([False, True, False, True])
    ranked = index.search("cat dog", k=10, allowed=allowed)
    assert set(ranked.positions.tolist()) <= {1, 3}


def test_changing_params_recomputes_weights() -> None:
    index = BM25Index.build(CORPUS, PLAIN, BM25Params(k1=1.2, b=0.75))
    before = index.score("pets")
    index.params = BM25Params(k1=0.5, b=0.0)
    after = index.score("pets")
    np.testing.assert_allclose(after, naive_bm25(CORPUS, "pets", 0.5, 0.0), rtol=1e-5)
    assert not np.allclose(before, after)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    index = BM25Index.build(CORPUS, TokenizerConfig(stemmer="light"), BM25Params(k1=1.1, b=0.6))
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert loaded.params == index.params
    assert loaded.tokenizer.config == index.tokenizer.config
    assert loaded.vocab == index.vocab
    np.testing.assert_allclose(loaded.score("cats chasing dogs"), index.score("cats chasing dogs"))


def test_params_validation() -> None:
    with pytest.raises(ValueError):
        BM25Params(b=1.5)
    with pytest.raises(ValueError):
        BM25Params(k1=-1)

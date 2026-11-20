"""Okapi BM25 built from scratch on SciPy sparse matrices.

For a query ``q`` and document ``d``::

    score(q, d) = sum over query terms t of  qtf(t) * idf(t) * tf(t, d) * (k1 + 1)
                                              ---------------------------------------------
                                              tf(t, d) + k1 * (1 - b + b * |d| / avgdl)

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))        (the Lucene variant, never negative)

The per-(document, term) part does not depend on the query, so it is computed once into a weight matrix
``W`` (documents x terms). Scoring is then a sparse matrix-vector product ``W @ q``. ``W`` is stored in CSC
format, where each column is exactly a posting list, so the product only touches documents that contain a
query term.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict, Field

from .ranking import top_k
from .schemas import Ranked
from .tokenizer import Tokenizer, TokenizerConfig


class BM25Params(BaseModel):
    """BM25 free parameters. ``k1`` controls term-frequency saturation, ``b`` document-length normalisation."""

    model_config = ConfigDict(frozen=True)

    k1: float = Field(default=0.9, ge=0.0)
    b: float = Field(default=0.4, ge=0.0, le=1.0)


class BM25Index:
    """A BM25 index over a fixed document collection.

    Build one with :meth:`build`; persist with :meth:`save` / :meth:`load`. Changing ``params`` re-derives the
    weight matrix from the stored term frequencies without re-tokenising the corpus.
    """

    def __init__(self, tf: sp.csr_matrix, vocab: dict[str, int], tokenizer_config: TokenizerConfig,
                 params: BM25Params | None = None) -> None:
        if tf.shape[1] != len(vocab):
            raise ValueError("term-frequency matrix width must equal the vocabulary size")
        self.tf = tf.tocsr().astype(np.float32)
        self.vocab = vocab
        self.tokenizer = Tokenizer(tokenizer_config)
        self.doc_len = np.asarray(self.tf.sum(axis=1)).ravel().astype(np.float32)
        n_docs = self.tf.shape[0]
        self.avgdl = float(self.doc_len.mean()) if n_docs else 0.0
        df = np.bincount(self.tf.indices, minlength=len(vocab)).astype(np.float64)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        self._params = params or BM25Params()
        self._weights = self._compute_weights()

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(cls, texts: Sequence[str], tokenizer_config: TokenizerConfig | None = None,
              params: BM25Params | None = None) -> BM25Index:
        """Tokenise ``texts`` and build the index (one row per text, in order)."""
        config = tokenizer_config or TokenizerConfig()
        tokenizer = Tokenizer(config)
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        for row, text in enumerate(texts):
            for token in tokenizer(text):
                rows.append(row)
                cols.append(vocab.setdefault(token, len(vocab)))
        data = np.ones(len(rows), dtype=np.float32)
        tf = sp.csr_matrix((data, (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
                           shape=(len(texts), len(vocab)), dtype=np.float32)
        tf.sum_duplicates()
        return cls(tf, vocab, config, params)

    def _compute_weights(self) -> sp.csc_matrix:
        k1, b = self._params.k1, self._params.b
        tf = self.tf
        row_of_entry = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
        avgdl = self.avgdl or 1.0
        norm = k1 * (1.0 - b + b * self.doc_len[row_of_entry] / avgdl)
        data = self.idf[tf.indices] * tf.data * (k1 + 1.0) / (tf.data + norm)
        return sp.csr_matrix((data.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape).tocsc()

    @property
    def params(self) -> BM25Params:
        return self._params

    @params.setter
    def params(self, value: BM25Params) -> None:
        self._params = value
        self._weights = self._compute_weights()

    @property
    def n_docs(self) -> int:
        return int(self.tf.shape[0])

    # ------------------------------------------------------------------ querying
    def query_vector(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """Known query term ids and their counts (unknown terms cannot match and are dropped)."""
        ids = [self.vocab[t] for t in self.tokenizer(query) if t in self.vocab]
        if not ids:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        unique, counts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
        return unique, counts.astype(np.float32)

    def score(self, query: str) -> np.ndarray:
        """BM25 score of every document for ``query`` (zeros where no query term occurs)."""
        term_ids, counts = self.query_vector(query)
        if term_ids.size == 0:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self._weights[:, term_ids] @ counts).ravel()

    def search(self, query: str, k: int = 10, allowed: np.ndarray | None = None) -> Ranked:
        """Top-``k`` documents with a positive score; ``allowed`` restricts the candidates (pre-filter)."""
        return top_k(self.score(query), k, allowed=allowed, min_score=0.0)

    # ------------------------------------------------------------------ persistence
    def save(self, folder: Path | str) -> None:
        """Write ``bm25_tf.npz`` (term frequencies) and ``bm25.json`` (vocabulary and settings)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        sp.save_npz(folder / "bm25_tf.npz", self.tf, compressed=True)
        terms = sorted(self.vocab, key=self.vocab.__getitem__)
        meta = {"params": self._params.model_dump(), "tokenizer": self.tokenizer.config.model_dump(), "terms": terms}
        (folder / "bm25.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path | str) -> BM25Index:
        """Inverse of :meth:`save`."""
        folder = Path(folder)
        meta = json.loads((folder / "bm25.json").read_text(encoding="utf-8"))
        tf = sp.load_npz(folder / "bm25_tf.npz").tocsr()
        vocab = {term: i for i, term in enumerate(meta["terms"])}
        return cls(tf, vocab, TokenizerConfig(**meta["tokenizer"]), BM25Params(**meta["params"]))

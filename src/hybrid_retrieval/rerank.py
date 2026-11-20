"""Second-stage rerankers that rescore a short candidate list with the query and document read together.

* :class:`CrossEncoderReranker` runs ``Xenova/ms-marco-MiniLM-L-6-v2`` through fastembed. A cross-encoder
  feeds ``[query, passage]`` jointly through a transformer, so it sees word-level interactions a bi-encoder
  cannot; the price is one forward pass per candidate, which is why it only rescores the top ``N``.
* :class:`TermOverlapReranker` is a deterministic, download-free stand-in for tests and offline demos. It scores
  query-term coverage plus adjacent query-bigram matches. It is not a learned model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .embedders import fastembed_cache_dir
from .schemas import Ranked
from .tokenizer import Tokenizer

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@runtime_checkable
class Reranker(Protocol):
    """Scores (query, passage) pairs; higher means more relevant."""

    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """One float per passage, in input order."""
        ...


class CrossEncoderReranker:
    """fastembed ``TextCrossEncoder`` with batched scoring."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, *, batch_size: int = 32,
                 local_files_only: bool = False, threads: int | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        kwargs: dict[str, Any] = {"cache_dir": fastembed_cache_dir(), "threads": threads}
        if local_files_only:
            kwargs["local_files_only"] = True
        self._model = TextCrossEncoder(model_name, **kwargs)
        self.model_name = model_name
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return f"cross-encoder:{self.model_name}"

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if len(passages) == 0:
            return np.zeros(0, dtype=np.float64)
        return np.fromiter(self._model.rerank(query, list(passages), batch_size=self.batch_size),
                           dtype=np.float64, count=len(passages))


class TermOverlapReranker:
    """Offline stand-in: share of distinct query terms present, plus half the share of query bigrams present."""

    def __init__(self) -> None:
        self._tokenizer = Tokenizer()

    @property
    def name(self) -> str:
        return "term-overlap (offline stand-in)"

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        q_tokens = self._tokenizer(query)
        q_terms = set(q_tokens)
        q_bigrams = set(zip(q_tokens, q_tokens[1:], strict=False))
        out = np.zeros(len(passages), dtype=np.float64)
        if not q_terms:
            return out
        for i, passage in enumerate(passages):
            tokens = self._tokenizer(passage)
            coverage = len(q_terms.intersection(tokens)) / len(q_terms)
            pairs = set(zip(tokens, tokens[1:], strict=False))
            bigram = len(q_bigrams & pairs) / len(q_bigrams) if q_bigrams else 0.0
            out[i] = coverage + 0.5 * bigram
        return out


def rerank_head(ranked: Ranked, depth: int, scores: np.ndarray) -> Ranked:
    """Reorder the first ``depth`` entries of ``ranked`` by ``scores`` (one per entry); keep the tail as it was.

    Keeping the tail means a reranked list stays as long as its input, so Recall@100 is still defined when only
    the top 20 were rescored. The tail keeps its first-stage order but is placed strictly below the head.
    """
    depth = min(depth, len(ranked))
    if scores.shape != (depth,):
        raise ValueError(f"expected {depth} scores, got {scores.shape}")
    head_pos = ranked.positions[:depth]
    order = np.lexsort((np.arange(depth), -scores))
    head_scores = scores[order]
    tail_pos = ranked.positions[depth:]
    floor = float(head_scores.min()) if depth else 0.0
    tail_scores = floor - 1.0 - np.arange(tail_pos.size, dtype=np.float64)
    return Ranked(np.concatenate([head_pos[order], tail_pos]).astype(np.int64),
                  np.concatenate([head_scores, tail_scores]))


def load_reranker(kind: str, *, local_files_only: bool = False) -> Reranker:
    """Build a reranker from a short name: ``"overlap"`` or ``"minilm"`` (the cross-encoder)."""
    if kind == "overlap":
        return TermOverlapReranker()
    if kind in ("minilm", "cross-encoder", DEFAULT_RERANK_MODEL):
        return CrossEncoderReranker(local_files_only=local_files_only)
    raise ValueError(f"unknown reranker {kind!r}; use 'overlap' or 'minilm'")

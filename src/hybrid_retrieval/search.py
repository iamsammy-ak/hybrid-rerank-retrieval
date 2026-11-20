"""The query pipeline: filter -> sparse + dense retrieval -> fusion -> optional cross-encoder rerank.

Each stage is a public method that returns its ranked list and its wall-clock time, so the benchmark harness
runs exactly the same code as :meth:`HybridSearcher.search` while caching stage outputs across a sweep.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .embedders import Embedder
from .filters import FilterExpr, coerce_filter, to_text
from .fusion import Normalisation, convex, rrf
from .index import HybridIndex
from .rerank import Reranker, rerank_head
from .schemas import Hit, Ranked, SearchResponse, StageOutput

Mode = Literal["bm25", "dense", "hybrid"]


class SearchConfig(BaseModel):
    """Every knob of the pipeline, with the defaults used by the CLI and the API."""

    model_config = ConfigDict(frozen=True)

    mode: Mode = "hybrid"
    fusion: Literal["rrf", "convex"] = "rrf"
    rrf_k: float = Field(default=60.0, gt=0.0)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight on the dense list in convex fusion.")
    norm: Normalisation = "minmax"
    candidates: int = Field(default=100, ge=1, le=10_000, description="Hits each retriever passes to fusion.")
    rerank_depth: int = Field(default=0, ge=0, le=1_000, description="Fused hits the reranker rescores; 0 = off.")
    top_k: int = Field(default=10, ge=1, le=1_000)
    filter_mode: Literal["pre", "post"] = "pre"

    @property
    def label(self) -> str:
        """Short human-readable name, e.g. ``hybrid-rrf(k=60)+rerank@50``."""
        base: str
        if self.mode != "hybrid":
            base = self.mode
        elif self.fusion == "rrf":
            base = f"hybrid-rrf(k={self.rrf_k:g})"
        else:
            base = f"hybrid-convex(a={self.alpha:g},{self.norm})"
        return base + (f"+rerank@{self.rerank_depth}" if self.rerank_depth else "")


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


class HybridSearcher:
    """Runs queries against a :class:`HybridIndex`.

    ``embedder`` is required for ``dense`` and ``hybrid`` modes and must match the one used at indexing time;
    ``reranker`` is required when ``rerank_depth > 0``.
    """

    def __init__(self, index: HybridIndex, embedder: Embedder | None = None, reranker: Reranker | None = None) -> None:
        if embedder is not None and index.dense is not None:
            index.check_embedder(embedder)
        self.index = index
        self.embedder = embedder
        self.reranker = reranker

    # ------------------------------------------------------------------ stages
    def filter_mask(self, expr: FilterExpr | None) -> tuple[np.ndarray | None, float]:
        """Compile a filter to a boolean mask over the corpus."""
        if expr is None:
            return None, 0.0
        start = time.perf_counter()
        mask = self.index.metadata.mask(expr)
        return mask, _ms(start)

    def sparse(self, query: str, k: int, allowed: np.ndarray | None = None) -> tuple[Ranked, float]:
        """BM25 top-``k`` (pre-filtered by ``allowed`` when given)."""
        start = time.perf_counter()
        ranked = self.index.bm25.search(query, k, allowed=allowed)
        return ranked, _ms(start)

    def dense(self, query: str, k: int, allowed: np.ndarray | None = None) -> tuple[Ranked, float]:
        """Embed the query and take the cosine top-``k``; the time includes query encoding."""
        if self.embedder is None or self.index.dense is None:
            raise ValueError("dense retrieval needs an index with embeddings and a matching embedder")
        start = time.perf_counter()
        ranked = self.index.dense.search(self.embedder.embed_query(query), k, allowed=allowed)
        return ranked, _ms(start)

    @staticmethod
    def fuse(sparse: Ranked, dense: Ranked, config: SearchConfig) -> tuple[Ranked, float]:
        """Combine the two lists with the configured fusion method."""
        start = time.perf_counter()
        if config.fusion == "rrf":
            fused = rrf([sparse, dense], k=config.rrf_k)
        else:
            fused = convex(sparse, dense, alpha=config.alpha, norm=config.norm)
        return fused, _ms(start)

    def rerank(self, query: str, ranked: Ranked, depth: int) -> tuple[Ranked, float]:
        """Rescore the first ``depth`` hits with the reranker; the rest keep their order below them."""
        if self.reranker is None:
            raise ValueError("rerank_depth > 0 needs a reranker")
        start = time.perf_counter()
        depth = min(depth, len(ranked))
        passages = [self.index.docs[int(p)].content for p in ranked.positions[:depth]]
        scores = self.reranker.score(query, passages)
        return rerank_head(ranked, depth, scores), _ms(start)

    @staticmethod
    def post_filter(ranked: Ranked, mask: np.ndarray) -> Ranked:
        """Drop hits that fail the filter (post-filtering)."""
        keep = mask[ranked.positions]
        return Ranked(ranked.positions[keep], ranked.scores[keep])

    # ------------------------------------------------------------------ public API
    def hits(self, ranked: Ranked, k: int) -> list[Hit]:
        """Convert the first ``k`` entries of a ranked list into :class:`Hit` objects."""
        out = []
        for rank, (pos, score) in enumerate(zip(ranked.positions[:k], ranked.scores[:k], strict=True), start=1):
            doc = self.index.docs[int(pos)]
            snippet = doc.text if len(doc.text) <= 180 else doc.text[:177].rstrip() + "..."
            out.append(Hit(rank=rank, doc_id=doc.id, score=float(score), title=doc.title, snippet=snippet,
                           metadata=dict(doc.metadata)))
        return out

    def search(self, query: str, config: SearchConfig | None = None,
               filter: str | FilterExpr | Mapping[str, object] | None = None, *, trace_k: int = 5) -> SearchResponse:
        """Run the full pipeline for one query and return the hits plus a per-stage trace."""
        config = config or SearchConfig()
        expr = coerce_filter(filter)
        mask, filter_ms = self.filter_mask(expr)
        pre = mask if config.filter_mode == "pre" else None
        stages: list[StageOutput] = []
        timings: dict[str, float] = {}
        if expr is not None:
            timings["filter"] = filter_ms

        def record(name: str, ranked: Ranked, ms: float) -> None:
            timings[name] = ms
            stages.append(StageOutput(name=name, latency_ms=ms, hits=self.hits(ranked, trace_k)))

        if config.mode in ("bm25", "hybrid"):
            sparse, ms = self.sparse(query, config.candidates, pre)
            record("sparse", sparse, ms)
        if config.mode in ("dense", "hybrid"):
            dense, ms = self.dense(query, config.candidates, pre)
            record("dense", dense, ms)
        if config.mode == "hybrid":
            ranked, ms = self.fuse(sparse, dense, config)
            record("fusion", ranked, ms)
        else:
            ranked = sparse if config.mode == "bm25" else dense
        if mask is not None and config.filter_mode == "post":
            start = time.perf_counter()
            ranked = self.post_filter(ranked, mask)
            record("post-filter", ranked, _ms(start))
        if config.rerank_depth > 0:
            ranked, ms = self.rerank(query, ranked, config.rerank_depth)
            record("rerank", ranked, ms)
        timings["total"] = sum(timings.values())
        return SearchResponse(
            query=query,
            hits=self.hits(ranked, config.top_k),
            stages=stages,
            timings_ms=timings,
            filter=None if expr is None else to_text(expr),
            filter_mode=config.filter_mode,
            matching_docs=None if mask is None else int(mask.sum()),
        )

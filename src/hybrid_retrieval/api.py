"""A small FastAPI service exposing ``/search`` over a loaded index (served on port 8702 by the CLI)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import Field

from .filters import FilterError, FilterExpr
from .schemas import SearchResponse
from .search import HybridSearcher, Mode, SearchConfig

DEFAULT_PORT = 8702


class SearchRequest(SearchConfig):
    """A query plus any :class:`SearchConfig` field; ``filter`` is filter text or its JSON expression form."""

    query: str = Field(min_length=1, max_length=2_000)
    filter: str | FilterExpr | None = None


def create_app(searcher: HybridSearcher) -> FastAPI:
    """Build the app around an already loaded searcher (models are loaded once, not per request)."""
    app = FastAPI(title="hybrid-search", version="0.1.0",
                  description="BM25 + dense retrieval, rank fusion, cross-encoder reranking and metadata filters.")

    def run(query: str, config: SearchConfig, filter: str | FilterExpr | None) -> SearchResponse:
        try:
            return searcher.search(query, config, filter=filter)
        except FilterError as exc:
            raise HTTPException(status_code=400, detail=f"filter error: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "documents": len(searcher.index),
                "embedder": searcher.embedder.name if searcher.embedder else None,
                "reranker": searcher.reranker.name if searcher.reranker else None}

    @app.get("/fields")
    def fields() -> dict[str, str]:
        """Metadata fields that filters can use, with their types."""
        return searcher.index.metadata.fields

    @app.post("/search", response_model=SearchResponse)
    def search_post(request: SearchRequest) -> SearchResponse:
        config = SearchConfig(**request.model_dump(exclude={"query", "filter"}))
        return run(request.query, config, request.filter)

    @app.get("/search", response_model=SearchResponse)
    def search_get(q: str = Query(min_length=1, max_length=2_000), top_k: int = Query(10, ge=1, le=100),
                   mode: Mode = "hybrid", rerank: int = Query(0, ge=0, le=200),
                   filter: str | None = None) -> SearchResponse:
        return run(q, SearchConfig(mode=mode, top_k=top_k, rerank_depth=rerank), filter)

    return app

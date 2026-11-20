"""Data contracts shared by the indexes, the searcher, the CLI and the HTTP API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

#: A metadata value. ``None`` means "missing" and never matches a filter clause.
MetaValue = str | int | float | bool | None


class Document(BaseModel):
    """One retrievable unit: an id, an optional title, the body text and flat metadata."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    title: str = ""
    text: str
    metadata: dict[str, MetaValue] = Field(default_factory=dict)

    @property
    def content(self) -> str:
        """Title and body joined, which is what both retrievers and the reranker see."""
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass(frozen=True)
class Ranked:
    """A ranked list of internal document positions with their scores (best first).

    Internal positions index into the index's document list; they are cheaper to pass around than
    string ids while a query is in flight.
    """

    positions: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        if self.positions.shape != self.scores.shape or self.positions.ndim != 1:
            raise ValueError("positions and scores must be 1-D arrays of the same length")

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def head(self, k: int) -> Ranked:
        """The first ``k`` entries."""
        return Ranked(self.positions[:k], self.scores[:k])

    @classmethod
    def empty(cls) -> Ranked:
        return cls(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))


class Hit(BaseModel):
    """A single search result as returned to callers."""

    rank: int = Field(ge=1)
    doc_id: str
    score: float
    title: str = ""
    snippet: str = ""
    metadata: dict[str, MetaValue] = Field(default_factory=dict)


class StageOutput(BaseModel):
    """What one pipeline stage produced: its top hits and how long it took."""

    name: str
    latency_ms: float
    hits: list[Hit]


class SearchResponse(BaseModel):
    """Final hits plus a per-stage trace, so a caller can see where each result came from."""

    query: str
    hits: list[Hit]
    stages: list[StageOutput]
    timings_ms: dict[str, float]
    filter: str | None = None
    filter_mode: str = "pre"
    matching_docs: int | None = Field(default=None, description="Documents that satisfy the filter, if any.")

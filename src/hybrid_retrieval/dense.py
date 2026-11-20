"""Exact (brute-force) cosine top-k over a matrix of unit-length embeddings, with an optional int8 variant.

With unit vectors, cosine similarity is a dot product, so a query is one matrix-vector product over the corpus
followed by an ``argpartition`` top-k. For a few thousand to a few hundred thousand documents this is fast,
exact and has no index to tune; approximate indexes (HNSW, IVF) only pay off well beyond that.

The int8 variant stores each dimension as ``round(x / s_j)`` with a per-dimension scale
``s_j = max_i |x_ij| / 127``. Scores are computed as ``codes @ (q * s)``, so the query side stays in float32.
It cuts embedding memory by 4x; the accuracy cost is measured in the benchmark, not assumed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .embedders import Embedder, Progress
from .ranking import top_k
from .schemas import Ranked

_CHUNK_ROWS = 8192


class DenseIndex:
    """Exact inner-product search over float32 or int8-quantised embeddings."""

    def __init__(self, vectors: np.ndarray, *, scale: np.ndarray | None = None) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2-D array")
        if scale is None and vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if scale is not None and (vectors.dtype != np.int8 or scale.shape != (vectors.shape[1],)):
            raise ValueError("a quantised index needs int8 codes and one float scale per dimension")
        self.vectors = vectors
        self.scale = None if scale is None else scale.astype(np.float32)

    @classmethod
    def build(cls, texts: Sequence[str], embedder: Embedder, progress: Progress | None = None) -> DenseIndex:
        """Embed ``texts`` with ``embedder`` (one row per text, in order)."""
        return cls(embedder.embed_documents(texts, progress=progress))

    @property
    def quantised(self) -> bool:
        return self.scale is not None

    @property
    def n_docs(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def nbytes(self) -> int:
        """Memory held by the embeddings (and scales), in bytes."""
        return int(self.vectors.nbytes + (0 if self.scale is None else self.scale.nbytes))

    def quantise_int8(self) -> DenseIndex:
        """Return an int8 copy with symmetric per-dimension scaling."""
        if self.quantised:
            return self
        max_abs = np.abs(self.vectors).max(axis=0)
        scale = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
        codes = np.clip(np.rint(self.vectors / scale), -127, 127).astype(np.int8)
        return DenseIndex(codes, scale=scale)

    def score(self, query_vector: np.ndarray) -> np.ndarray:
        """Similarity of every document to ``query_vector``."""
        q = np.asarray(query_vector, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"query vector has shape {q.shape}, expected ({self.dim},)")
        if self.scale is None:
            return self.vectors @ q
        q_scaled = q * self.scale
        out = np.empty(self.n_docs, dtype=np.float32)
        for start in range(0, self.n_docs, _CHUNK_ROWS):
            block = self.vectors[start : start + _CHUNK_ROWS]
            out[start : start + block.shape[0]] = block.astype(np.float32) @ q_scaled
        return out

    def search(self, query_vector: np.ndarray, k: int = 10, allowed: np.ndarray | None = None) -> Ranked:
        """Top-``k`` documents by cosine similarity; ``allowed`` restricts the candidates (pre-filter)."""
        return top_k(self.score(query_vector), k, allowed=allowed)

    def save(self, folder: Path | str) -> None:
        """Write ``dense.npy`` (float32) or ``dense_int8.npz`` (codes and scales)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if self.scale is None:
            np.save(folder / "dense.npy", self.vectors)
        else:
            np.savez(folder / "dense_int8.npz", codes=self.vectors, scale=self.scale)

    @classmethod
    def load(cls, folder: Path | str, *, quantised: bool = False) -> DenseIndex:
        """Inverse of :meth:`save`."""
        folder = Path(folder)
        if quantised:
            with np.load(folder / "dense_int8.npz") as data:
                return cls(data["codes"], scale=data["scale"])
        return cls(np.load(folder / "dense.npy"))

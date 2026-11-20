"""Text embedders for the dense retriever.

* :class:`FastEmbedEmbedder` runs ``BAAI/bge-small-en-v1.5`` (384 dimensions) through fastembed's ONNX runtime.
  BGE v1.5 expects short retrieval queries to carry the instruction
  ``"Represent this sentence for searching relevant passages: "`` and passages to carry no prefix; both
  prefixes are configurable and recorded in the index manifest.
* :class:`HashingEmbedder` is a deterministic, download-free stand-in for tests and offline demos: signed
  feature hashing of word unigrams and character n-grams, L2-normalised. It captures surface similarity
  (shared words and word pieces), not meaning.
"""

from __future__ import annotations

import copy
import os
import zlib
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .tokenizer import Tokenizer, TokenizerConfig

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

#: ``progress(done, total)`` callback used while embedding a corpus.
Progress = Callable[[int, int], None]


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps text to L2-normalised float32 vectors."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str], progress: Progress | None = None) -> np.ndarray:
        """Embed passages; returns a ``(len(texts), dim)`` float32 array with unit-length rows.

        ``progress(done, total)``, if given, is called as batches complete.
        """
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query; returns a ``(dim,)`` float32 unit vector."""
        ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.where(norms == 0.0, 1.0, norms)).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing of words and character n-grams (an offline stand-in, not a semantic model).

    Each feature is hashed with CRC-32 (stable across runs and platforms, unlike Python's ``hash``); the low bits
    choose the dimension and bit 31 the sign, which keeps collisions unbiased. Counts are log-scaled and the
    vector is L2-normalised, so the dot product is a cosine similarity.
    """

    def __init__(self, dim: int = 384, ngram_range: tuple[int, int] = (3, 5), char_weight: float = 0.5) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        if not 1 <= ngram_range[0] <= ngram_range[1]:
            raise ValueError("ngram_range must be (min, max) with 1 <= min <= max")
        self._dim = dim
        self.ngram_range = ngram_range
        self.char_weight = char_weight
        self._tokenizer = Tokenizer(TokenizerConfig(stemmer="light"))

    @property
    def name(self) -> str:
        lo, hi = self.ngram_range
        return f"hashing-{self._dim}-c{lo}{hi}"

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> dict[str, float]:
        feats: dict[str, float] = {}
        lo, hi = self.ngram_range
        for token in self._tokenizer(text):
            feats["w:" + token] = feats.get("w:" + token, 0.0) + 1.0
            padded = f"#{token}#"
            for n in range(lo, hi + 1):
                for i in range(len(padded) - n + 1):
                    key = "c:" + padded[i : i + n]
                    feats[key] = feats.get(key, 0.0) + self.char_weight
        return feats

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float64)
        for feature, count in self._features(text).items():
            h = zlib.crc32(feature.encode("utf-8"))
            sign = -1.0 if h & 0x80000000 else 1.0
            vec[h % self._dim] += sign * np.log1p(count)
        return vec

    def embed_documents(self, texts: Sequence[str], progress: Progress | None = None) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = _l2_normalise(np.vstack([self._vector(t) for t in texts]))
        if progress is not None:
            progress(len(texts), len(texts))
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return _l2_normalise(self._vector(text)[None, :])[0]


def fastembed_cache_dir() -> str | None:
    """The model cache folder from ``FASTEMBED_CACHE_PATH``, if set."""
    return os.environ.get("FASTEMBED_CACHE_PATH") or None


class FastEmbedEmbedder:
    """``BAAI/bge-small-en-v1.5`` (or another fastembed model) with explicit query/passage prefixes."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, *, query_prefix: str = BGE_QUERY_INSTRUCTION,
                 passage_prefix: str = "", batch_size: int = 16, local_files_only: bool = False,
                 threads: int | None = None) -> None:
        from fastembed import TextEmbedding

        kwargs: dict[str, Any] = {"cache_dir": fastembed_cache_dir(), "threads": threads}
        if local_files_only:
            kwargs["local_files_only"] = True
        self._model = TextEmbedding(model_name, **kwargs)
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self._dim = int(self._model.embedding_size)

    def with_prefixes(self, *, query_prefix: str | None = None, passage_prefix: str | None = None) -> FastEmbedEmbedder:
        """A copy with different prefixes that shares the already loaded model (no second ONNX session)."""
        clone = copy.copy(self)
        if query_prefix is not None:
            clone.query_prefix = query_prefix
        if passage_prefix is not None:
            clone.passage_prefix = passage_prefix
        return clone

    @property
    def name(self) -> str:
        return f"fastembed:{self.model_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str], progress: Progress | None = None) -> np.ndarray:
        """Embed passages in length-sorted batches, so each batch pads to similar lengths, then restore order.

        A transformer batch is padded to its longest member; sorting by length first avoids paying for 512-token
        padding on short abstracts. ``progress(done, total)`` is called after each batch.
        """
        if len(texts) == 0:
            return np.zeros((0, self._dim), dtype=np.float32)
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.empty((len(texts), self._dim), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            batch = order[start : start + self.batch_size]
            chunk = [self.passage_prefix + texts[i] for i in batch]
            out[batch] = np.vstack(list(self._model.embed(chunk, batch_size=self.batch_size)))
            if progress is not None:
                progress(min(start + self.batch_size, len(texts)), len(texts))
        return _l2_normalise(out)

    def embed_query(self, text: str) -> np.ndarray:
        vec = next(iter(self._model.embed([self.query_prefix + text], batch_size=1)))
        return _l2_normalise(np.asarray(vec, dtype=np.float32)[None, :])[0]


def load_embedder(kind: str, *, local_files_only: bool = False) -> Embedder:
    """Build an embedder from a short name: ``"hashing"`` or ``"bge-small"`` (alias of the fastembed default)."""
    if kind == "hashing":
        return HashingEmbedder()
    if kind in ("bge-small", "fastembed", DEFAULT_EMBED_MODEL):
        return FastEmbedEmbedder(local_files_only=local_files_only)
    raise ValueError(f"unknown embedder {kind!r}; use 'hashing' or 'bge-small'")

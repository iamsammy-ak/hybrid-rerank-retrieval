"""A persisted hybrid index: documents, a BM25 index, dense embeddings and a metadata table, side by side.

On-disk layout of an index folder::

    manifest.json     format version, document count, embedder identity, tokenizer and BM25 settings
    docs.jsonl        the documents (id, title, text, metadata), in index order
    bm25_tf.npz       sparse term-frequency matrix         bm25.json   vocabulary and BM25 settings
    dense.npy         float32 embeddings (unit rows)       dense_int8.npz   optional int8 codes + scales

Loading checks that the embedder used for queries is the one that produced the stored embeddings, because
mixing embedding spaces fails silently otherwise (the scores look plausible and are meaningless).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .bm25 import BM25Index, BM25Params
from .dense import DenseIndex
from .embedders import Embedder, FastEmbedEmbedder, HashingEmbedder, Progress
from .filters import MetadataTable
from .schemas import Document
from .tokenizer import TokenizerConfig

FORMAT_VERSION = 1


class EmbedderInfo(BaseModel):
    """Identity of the embedder that produced the stored vectors."""

    name: str
    dim: int
    passage_prefix: str = ""


def describe_embedder(embedder: Embedder) -> EmbedderInfo:
    """Capture what must match between indexing time and query time."""
    prefix = embedder.passage_prefix if isinstance(embedder, FastEmbedEmbedder) else ""
    return EmbedderInfo(name=embedder.name, dim=embedder.dim, passage_prefix=prefix)


def embedder_from_info(info: EmbedderInfo, *, local_files_only: bool = False) -> Embedder:
    """Recreate the query-side embedder recorded in an index manifest."""
    if info.name.startswith("fastembed:"):
        return FastEmbedEmbedder(info.name.split(":", 1)[1], passage_prefix=info.passage_prefix,
                                 local_files_only=local_files_only)
    hashing = HashingEmbedder(dim=info.dim)
    if info.name == hashing.name:
        return hashing
    raise ValueError(f"cannot recreate embedder {info.name!r}")


class HybridIndex:
    """Everything needed to answer a hybrid query over one document collection."""

    def __init__(self, docs: Sequence[Document], bm25: BM25Index, dense: DenseIndex | None,
                 embedder_info: EmbedderInfo | None) -> None:
        if bm25.n_docs != len(docs) or (dense is not None and dense.n_docs != len(docs)):
            raise ValueError("BM25 rows, dense rows and documents must line up one to one")
        ids = [d.id for d in docs]
        if len(set(ids)) != len(ids):
            raise ValueError("document ids must be unique")
        self.docs = list(docs)
        self.bm25 = bm25
        self.dense = dense
        self.embedder_info = embedder_info
        self.metadata = MetadataTable([d.metadata for d in self.docs])
        self.position = {doc_id: i for i, doc_id in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.docs)

    @classmethod
    def build(cls, docs: Sequence[Document], embedder: Embedder | None, *,
              tokenizer_config: TokenizerConfig | None = None, bm25_params: BM25Params | None = None,
              progress: Progress | None = None) -> HybridIndex:
        """Index ``docs`` for BM25 and, if an embedder is given, for dense retrieval."""
        texts = [d.content for d in docs]
        bm25 = BM25Index.build(texts, tokenizer_config, bm25_params)
        dense = DenseIndex.build(texts, embedder, progress) if embedder is not None else None
        info = describe_embedder(embedder) if embedder is not None else None
        return cls(docs, bm25, dense, info)

    def check_embedder(self, embedder: Embedder) -> None:
        """Raise if ``embedder`` is not the one the stored vectors came from."""
        if self.embedder_info is None:
            raise ValueError("this index has no dense embeddings")
        actual = describe_embedder(embedder)
        if (actual.name, actual.dim, actual.passage_prefix) != (
            self.embedder_info.name, self.embedder_info.dim, self.embedder_info.passage_prefix):
            raise ValueError(f"index was built with {self.embedder_info.name} (dim {self.embedder_info.dim}); "
                             f"got {actual.name} (dim {actual.dim})")

    def save(self, folder: Path | str, *, with_int8: bool = False) -> None:
        """Write the index folder (see the module docstring for the layout)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "docs.jsonl").open("w", encoding="utf-8") as fh:
            for doc in self.docs:
                fh.write(doc.model_dump_json() + "\n")
        self.bm25.save(folder)
        if self.dense is not None:
            self.dense.save(folder)
            if with_int8:
                self.dense.quantise_int8().save(folder)
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "n_docs": len(self.docs),
            "embedder": None if self.embedder_info is None else self.embedder_info.model_dump(),
            "tokenizer": self.bm25.tokenizer.config.model_dump(),
            "bm25": self.bm25.params.model_dump(),
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path | str, *, quantised: bool = False) -> HybridIndex:
        """Read an index folder; ``quantised=True`` loads the int8 embeddings instead of float32."""
        folder = Path(folder)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported index format {manifest.get('format_version')!r}")
        with (folder / "docs.jsonl").open(encoding="utf-8") as fh:
            docs = [Document.model_validate_json(line) for line in fh if line.strip()]
        bm25 = BM25Index.load(folder)
        info = EmbedderInfo(**manifest["embedder"]) if manifest.get("embedder") else None
        dense = DenseIndex.load(folder, quantised=quantised) if info is not None else None
        return cls(docs, bm25, dense, info)

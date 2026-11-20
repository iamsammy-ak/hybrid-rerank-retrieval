"""Hybrid BM25 + dense retrieval with rank fusion, cross-encoder reranking and typed metadata filters."""

from .bm25 import BM25Index, BM25Params
from .dense import DenseIndex
from .embedders import Embedder, FastEmbedEmbedder, HashingEmbedder
from .filters import FilterError, MetadataTable, parse_filter
from .fusion import convex, rrf
from .index import HybridIndex
from .metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k
from .rerank import CrossEncoderReranker, Reranker, TermOverlapReranker
from .schemas import Document, Hit, SearchResponse
from .search import HybridSearcher, SearchConfig
from .tokenizer import Tokenizer, TokenizerConfig

__version__ = "0.1.0"

__all__ = [
    "BM25Index",
    "BM25Params",
    "CrossEncoderReranker",
    "DenseIndex",
    "Document",
    "Embedder",
    "FastEmbedEmbedder",
    "FilterError",
    "HashingEmbedder",
    "Hit",
    "HybridIndex",
    "HybridSearcher",
    "MetadataTable",
    "Reranker",
    "SearchConfig",
    "SearchResponse",
    "TermOverlapReranker",
    "Tokenizer",
    "TokenizerConfig",
    "__version__",
    "convex",
    "evaluate",
    "mrr_at_k",
    "ndcg_at_k",
    "parse_filter",
    "recall_at_k",
    "rrf",
]

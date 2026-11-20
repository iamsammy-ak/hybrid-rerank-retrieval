from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hybrid_retrieval.embedders import HashingEmbedder
from hybrid_retrieval.filters import FilterError, parse_filter
from hybrid_retrieval.index import HybridIndex, embedder_from_info
from hybrid_retrieval.rerank import TermOverlapReranker
from hybrid_retrieval.schemas import Document
from hybrid_retrieval.search import HybridSearcher, SearchConfig


def test_demo_corpus_shape(demo_docs: list[Document]) -> None:
    assert len(demo_docs) == 48
    assert {d.metadata["category"] for d in demo_docs} == {"databases", "networking", "security", "ml", "devops",
                                                           "frontend"}
    assert all(isinstance(d.metadata["year"], int) for d in demo_docs)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_every_mode_returns_ranked_hits(searcher: HybridSearcher, mode: str) -> None:
    resp = searcher.search("rotate a leaked API key", SearchConfig(mode=mode, top_k=5))  # type: ignore[arg-type]
    assert 0 < len(resp.hits) <= 5
    assert [h.rank for h in resp.hits] == list(range(1, len(resp.hits) + 1))
    assert resp.hits[0].doc_id == "kb-sec-01"
    assert "total" in resp.timings_ms


def test_stage_trace(searcher: HybridSearcher) -> None:
    resp = searcher.search("E1047", SearchConfig(rerank_depth=10), trace_k=3)
    assert [s.name for s in resp.stages] == ["sparse", "dense", "fusion", "rerank"]
    assert all(len(s.hits) <= 3 for s in resp.stages)
    assert resp.stages[0].hits[0].doc_id == "kb-db-07"


def test_convex_mode(searcher: HybridSearcher) -> None:
    resp = searcher.search("canary rollback", SearchConfig(fusion="convex", alpha=0.3, norm="zscore"))
    assert resp.hits[0].doc_id == "kb-ops-07"


def test_pre_filter_only_returns_matches_and_fills_top_k(searcher: HybridSearcher) -> None:
    flt = 'category == "security" or year <= 2018'
    resp = searcher.search("database performance", SearchConfig(top_k=8, filter_mode="pre"), filter=flt)
    mask = searcher.index.metadata.mask(parse_filter(flt))
    assert resp.matching_docs == int(mask.sum())
    assert len(resp.hits) == 8
    for hit in resp.hits:
        assert hit.metadata["category"] == "security" or hit.metadata["year"] <= 2018  # type: ignore[operator]


def test_post_filter_can_return_fewer_hits(searcher: HybridSearcher) -> None:
    flt = 'category == "frontend"'
    pre = searcher.search("database connection pooling", SearchConfig(candidates=5, filter_mode="pre"), filter=flt)
    post = searcher.search("database connection pooling", SearchConfig(candidates=5, filter_mode="post"), filter=flt)
    assert len(pre.hits) == 5  # each retriever still returns 5 matching documents
    assert all(h.metadata["category"] == "frontend" for h in pre.hits)
    assert len(post.hits) < len(pre.hits)
    assert all(h.metadata["category"] == "frontend" for h in post.hits)
    assert "post-filter" in [s.name for s in post.stages]


def test_filter_matching_nothing(searcher: HybridSearcher) -> None:
    resp = searcher.search("anything", filter="year > 3000")
    assert resp.hits == [] and resp.matching_docs == 0


def test_bad_filter_raises(searcher: HybridSearcher) -> None:
    with pytest.raises(FilterError):
        searcher.search("x", filter="colour == red")


def test_rerank_keeps_list_length(searcher: HybridSearcher) -> None:
    base = searcher.search("slow page load", SearchConfig(top_k=30))
    reranked = searcher.search("slow page load", SearchConfig(top_k=30, rerank_depth=5))
    assert len(base.hits) == len(reranked.hits)
    assert {h.doc_id for h in base.hits[:5]} == {h.doc_id for h in reranked.hits[:5]}
    assert [h.doc_id for h in base.hits[5:]] == [h.doc_id for h in reranked.hits[5:]]


def test_missing_components_raise(demo_index: HybridIndex) -> None:
    bare = HybridSearcher(demo_index)
    assert bare.search("ssh", SearchConfig(mode="bm25")).hits
    with pytest.raises(ValueError, match="embedder"):
        bare.search("ssh", SearchConfig(mode="dense"))
    with pytest.raises(ValueError, match="reranker"):
        bare.search("ssh", SearchConfig(mode="bm25", rerank_depth=3))


def test_embedder_mismatch_is_rejected(demo_index: HybridIndex) -> None:
    with pytest.raises(ValueError, match="built with"):
        HybridSearcher(demo_index, HashingEmbedder(dim=128))


def test_config_labels_and_validation() -> None:
    assert SearchConfig().label == "hybrid-rrf(k=60)"
    assert SearchConfig(fusion="convex", alpha=0.3, rerank_depth=20).label == "hybrid-convex(a=0.3,minmax)+rerank@20"
    assert SearchConfig(mode="bm25").label == "bm25"
    with pytest.raises(ValueError):
        SearchConfig(alpha=2)
    with pytest.raises(ValueError):
        SearchConfig(top_k=0)


def test_index_save_load_roundtrip(tmp_path: Path, demo_docs: list[Document]) -> None:
    built = HybridIndex.build(demo_docs, HashingEmbedder())
    built.save(tmp_path, with_int8=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_docs"] == 48 and manifest["embedder"]["name"] == HashingEmbedder().name
    loaded = HybridIndex.load(tmp_path)
    assert loaded.embedder_info is not None
    embedder = embedder_from_info(loaded.embedder_info)
    a = HybridSearcher(built, HashingEmbedder(), TermOverlapReranker()).search("ssh keys", SearchConfig(rerank_depth=5))
    b = HybridSearcher(loaded, embedder, TermOverlapReranker()).search("ssh keys", SearchConfig(rerank_depth=5))
    assert [h.doc_id for h in a.hits] == [h.doc_id for h in b.hits]
    quant = HybridIndex.load(tmp_path, quantised=True)
    assert quant.dense is not None and quant.dense.quantised
    assert loaded.docs == demo_docs


def test_bm25_only_index(tmp_path: Path, demo_docs: list[Document]) -> None:
    HybridIndex.build(demo_docs, None).save(tmp_path)
    loaded = HybridIndex.load(tmp_path)
    assert loaded.dense is None and loaded.embedder_info is None
    with pytest.raises(ValueError, match="no dense"):
        loaded.check_embedder(HashingEmbedder())


def test_index_rejects_bad_input(tmp_path: Path, demo_docs: list[Document]) -> None:
    with pytest.raises(ValueError, match="unique"):
        HybridIndex.build([demo_docs[0], demo_docs[0]], None)
    HybridIndex.build(demo_docs[:3], None).save(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace('"format_version": 1', '"format_version": 99'),
                        encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        HybridIndex.load(tmp_path)


def test_document_content() -> None:
    assert Document(id="a", title="T", text="body").content == "T\nbody"
    assert Document(id="a", text="body").content == "body"
    with pytest.raises(ValueError):
        Document(id="", text="x")


def test_hits_carry_metadata_and_snippets(searcher: HybridSearcher) -> None:
    hit = searcher.search("IPv6", SearchConfig(top_k=1)).hits[0]
    assert hit.doc_id == "kb-net-08" and hit.metadata["year"] == 2025
    assert len(hit.snippet) <= 180
    assert np.isfinite(hit.score)

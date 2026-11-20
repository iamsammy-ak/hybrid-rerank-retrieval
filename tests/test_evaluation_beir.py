from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hybrid_retrieval import beir
from hybrid_retrieval.beir import BeirDataset, download_beir, load_beir
from hybrid_retrieval.evaluation import BenchSettings, pareto_frontier, run_benchmark
from hybrid_retrieval.report import to_markdown
from hybrid_retrieval.schemas import Document
from hybrid_retrieval.search import HybridSearcher


def test_pareto_hand_example() -> None:
    points = [(10.0, 0.60), (20.0, 0.70), (15.0, 0.55), (30.0, 0.70), (5.0, 0.40), (40.0, 0.75)]
    assert pareto_frontier(points) == [4, 0, 1, 5]


@given(st.lists(st.tuples(st.floats(0, 100), st.floats(0, 1)), min_size=1, max_size=30))
def test_pareto_properties(points: list[tuple[float, float]]) -> None:
    front = set(pareto_frontier(points))

    def dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return a[0] <= b[0] and a[1] >= b[1] and a != b

    for i in front:  # nothing dominates a frontier point
        assert not any(dominates(points[j], points[i]) for j in range(len(points)) if j != i)
    for i in set(range(len(points))) - front:  # everything else is dominated or duplicates a frontier point
        assert any(dominates(points[j], points[i]) or points[j] == points[i] for j in front)


def test_load_beir(tiny_beir: Path) -> None:
    ds = load_beir(tiny_beir)
    assert [d.id for d in ds.corpus] == ["d1", "d2", "d3"]
    assert ds.corpus[1].metadata == {}  # nested metadata is dropped
    assert set(ds.queries) == {"q1", "q2"}  # q3 has no judgements
    assert ds.qrels == {"q1": {"d1": 1}, "q2": {"d2": 2}}  # zero grades dropped


def test_qrels_header_is_checked(tiny_beir: Path) -> None:
    (tiny_beir / "qrels" / "test.tsv").write_text("a\tb\tc\nq1\td1\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_beir(tiny_beir)


def test_download_reuses_existing_folder(tiny_beir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(beir, "_fetch", lambda *a, **k: pytest.fail("network used"))
    folder, source = download_beir("tiny", tiny_beir.parent)
    assert folder == tiny_beir and source == "cache"


def test_download_falls_back_to_mirror_and_checks_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch(url: str, dest: Path, timeout: float) -> None:
        if "ukp" in url:
            raise OSError("unreachable")
        dest.write_bytes(b"not the real file")

    monkeypatch.setattr(beir, "_fetch", fake_fetch)
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_beir("scifact", tmp_path)
    assert not any(tmp_path.rglob("*.part"))
    with pytest.raises(ValueError, match="no pinned"):
        download_beir("unknown-set", tmp_path)


def test_safe_extract_rejects_zip_slip(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../outside.txt", "nope")
    with pytest.raises(ValueError, match="unsafe"):
        beir._safe_extract(archive, tmp_path / "out")


def test_benchmark_end_to_end_on_demo_corpus(searcher: HybridSearcher, demo_docs: list[Document]) -> None:
    queries = {"q1": "rotate leaked api key", "q2": "pods restarting crash loop", "q3": "dns ttl propagation",
               "q4": "int8 quantised embeddings"}
    qrels = {"q1": {"kb-sec-01": 1}, "q2": {"kb-ops-01": 1}, "q3": {"kb-net-02": 1}, "q4": {"kb-ml-05": 2}}
    ds = BeirDataset("demo", demo_docs, queries, qrels)
    settings = BenchSettings(rrf_ks=[10, 60], alphas=[0.0, 1.0], rerank_depths=[5, 10], rerank_step=5,
                             headline_depth=5, filter_selectivities=[0.5, 0.01], warmup_queries=1)
    result = run_benchmark(ds, searcher, settings, int8=searcher.index.dense.quantise_int8())  # type: ignore[union-attr]
    names = [m["name"] for m in result["methods"]]
    assert names[:3] == ["BM25", "Dense", "Dense int8"]
    assert "Hybrid RRF + rerank@5" in names and "BM25 + rerank@5" in names
    assert result["n_queries"] == 4 and result["dense_memory"]["int8_bytes"] < result["dense_memory"]["float32_bytes"]
    assert len(result["sweeps"]["rrf_k"]) == 2 and len(result["rerank_sweep"]) == 2
    assert {r["mode"] for r in result["filtering"]} == {"pre", "post"}
    tiny = [r for r in result["filtering"] if r["selectivity"] == 0.01]
    assert tiny and all(r["judged_queries"] == 0 and r["metrics"] is None for r in tiny)  # nothing judged survives
    assert all(r["queries"] == 4 for r in result["filtering"])
    assert result["pareto_frontier"] and set(result["pareto_frontier"]) <= {p["name"] for p in result["all_points"]}
    rerank = next(m for m in result["methods"] if m["name"] == "Hybrid RRF + rerank@5")
    assert set(rerank["latency_ms"]) == {"sparse", "dense", "fusion", "rerank", "total"}
    bm25 = result["methods"][0]
    assert bm25["metrics"]["recall@100"] == 1.0
    markdown = to_markdown(result, "python -m hybrid_retrieval bench --dataset demo")
    assert "## Methods" in markdown and "Pre- vs post-filtering" in markdown and "| BM25 |" in markdown


def test_max_queries_limits_the_run(searcher: HybridSearcher, demo_docs: list[Document]) -> None:
    ds = BeirDataset("demo", demo_docs, {"a": "ssh", "b": "dns"}, {"a": {"kb-sec-04": 1}, "b": {"kb-net-02": 1}})
    result = run_benchmark(ds, searcher, BenchSettings(max_queries=1, rerank_depths=[], warmup_queries=0),
                           sweeps=False, filters=False)
    assert result["n_queries"] == 1 and result["sweeps"] == {} and result["filtering"] == []


def test_rerank_depths_must_be_multiples_of_the_step(searcher: HybridSearcher, demo_docs: list[Document]) -> None:
    ds = BeirDataset("demo", demo_docs, {"a": "ssh"}, {"a": {"kb-sec-04": 1}})
    with pytest.raises(ValueError, match="multiples"):
        run_benchmark(ds, searcher, BenchSettings(rerank_depths=[15], warmup_queries=0), sweeps=False, filters=False)


def test_sparse_benchmark(demo_docs: list[Document]) -> None:
    from hybrid_retrieval.evaluation import run_sparse_benchmark
    from hybrid_retrieval.report import sparse_to_markdown

    ds = BeirDataset("demo", demo_docs, {"a": "hardening ssh access", "b": "dns ttl"},
                     {"a": {"kb-sec-04": 1}, "b": {"kb-net-02": 1}})
    result = run_sparse_benchmark(ds, k1_grid=(0.9, 1.2), b_grid=(0.4, 0.75))
    assert [m["name"] for m in result["methods"]] == ["BM25", "BM25, no stemming"]
    assert result["methods"][0]["metrics"]["ndcg@10"] == 1.0
    assert len(result["grid"]) == 4
    markdown = sparse_to_markdown(result, "python -m hybrid_retrieval bench --sparse-only")
    assert "| k1 / b | 0.4 | 0.75 |" in markdown and "| BM25 |" in markdown


class CountingReranker:
    """Wraps the term-overlap stand-in and counts scored passages."""

    def __init__(self) -> None:
        from hybrid_retrieval.rerank import TermOverlapReranker

        self.inner = TermOverlapReranker()
        self.calls = 0

    @property
    def name(self) -> str:
        return "counting"

    def score(self, query: str, passages: list[str]) -> object:
        self.calls += len(passages)
        return self.inner.score(query, passages)


def test_rerank_scores_resume_from_checkpoint(demo_index: object, demo_docs: list[Document], tmp_path: Path) -> None:
    from hybrid_retrieval.embedders import HashingEmbedder

    ds = BeirDataset("demo", demo_docs, {"a": "ssh keys", "b": "dns ttl", "c": "canary rollback"},
                     {"a": {"kb-sec-04": 1}, "b": {"kb-net-02": 1}, "c": {"kb-ops-07": 1}})
    settings = BenchSettings(rerank_depths=[5, 10], rerank_step=5, headline_depth=5, warmup_queries=0,
                             bm25_rerank=False)
    first = CountingReranker()
    a = run_benchmark(ds, HybridSearcher(demo_index, HashingEmbedder(), first), settings,  # type: ignore[arg-type]
                      sweeps=False, filters=False, checkpoint_dir=tmp_path)
    assert first.calls == 30 and list(tmp_path.glob("rerank_*.json"))
    second = CountingReranker()
    b = run_benchmark(ds, HybridSearcher(demo_index, HashingEmbedder(), second), settings,  # type: ignore[arg-type]
                      sweeps=False, filters=False, checkpoint_dir=tmp_path)
    assert second.calls == 0  # everything came from the checkpoint
    assert a["rerank_sweep"][1]["metrics"] == b["rerank_sweep"][1]["metrics"]
    # scoring times come from the checkpoint; only the (sub-millisecond) reorder is timed again
    assert b["rerank_sweep"][1]["latency_ms"]["rerank"]["p50"] == pytest.approx(
        a["rerank_sweep"][1]["latency_ms"]["rerank"]["p50"], abs=0.5)


def test_with_prefixes_shares_the_model() -> None:
    from hybrid_retrieval.embedders import FastEmbedEmbedder

    emb = FastEmbedEmbedder.__new__(FastEmbedEmbedder)
    emb._model = object()  # type: ignore[assignment]
    emb.query_prefix, emb.passage_prefix = "q: ", "p: "
    clone = emb.with_prefixes(query_prefix="")
    assert clone._model is emb._model and clone.query_prefix == "" and clone.passage_prefix == "p: "
    assert emb.query_prefix == "q: "

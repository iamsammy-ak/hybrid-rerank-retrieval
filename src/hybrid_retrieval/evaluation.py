"""Benchmark harness: quality and per-stage latency for every pipeline variant, sweeps and a Pareto frontier.

How latency is measured: each query runs through the same stage methods as :meth:`HybridSearcher.search`,
sequentially, in one process, and each stage is timed with ``time.perf_counter``. A configuration's end-to-end
latency for a query is the sum of its stage times for that query; p50 and p95 are taken over queries. Sparse and
dense retrieval run once per query and are shared by all fusion variants (they do not depend on the fusion
settings). The cross-encoder scores each query's candidates in timed batches of ``rerank_step`` passages; the
rerank latency at depth ``N`` is the measured time of the first ``N / rerank_step`` batches, which is the work a
depth-``N`` call does, so every depth gets a measured latency from a single scoring pass.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from .beir import BeirDataset
from .bm25 import BM25Index, BM25Params
from .dense import DenseIndex
from .filters import Range
from .index import HybridIndex
from .metrics import evaluate, latency_summary
from .rerank import rerank_head
from .schemas import Ranked
from .search import HybridSearcher, SearchConfig
from .tokenizer import TokenizerConfig

Log = Callable[[str], None]


class BenchSettings(BaseModel):
    """What the benchmark runs. Defaults reproduce the committed SciFact results."""

    candidates: int = Field(default=100, ge=1)
    rrf_k: float = 60.0
    alpha: float = 0.5
    rrf_ks: list[float] = [1, 5, 10, 20, 40, 60, 100, 200]
    alphas: list[float] = [round(0.1 * i, 1) for i in range(11)]
    rerank_depths: list[int] = [10, 20, 30, 50]
    rerank_step: int = Field(default=10, ge=1)
    headline_depth: int = 50
    bm25_rerank: bool = True
    filter_selectivities: list[float] = [0.5, 0.1, 0.02]
    max_queries: int | None = None
    warmup_queries: int = 5
    seed: int = 13


@dataclass
class MethodResult:
    """Quality and latency of one pipeline configuration."""

    name: str
    family: str
    params: dict[str, Any]
    metrics: dict[str, float]
    stage_ms: dict[str, list[float]] = field(repr=False)

    @property
    def total_ms(self) -> list[float]:
        stages = list(self.stage_ms.values())
        return [float(sum(parts)) for parts in zip(*stages, strict=True)] if stages else []

    def to_json(self) -> dict[str, Any]:
        latency = {stage: latency_summary(ms) for stage, ms in self.stage_ms.items()}
        latency["total"] = latency_summary(self.total_ms)
        return {"name": self.name, "family": self.family, "params": self.params, "metrics": self.metrics,
                "latency_ms": latency}


def pareto_frontier(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the non-dominated points when lower latency (x) and higher quality (y) are both better.

    A point is kept if no other point is at least as fast and at least as good while strictly better in one of
    the two. Returned in order of increasing latency.
    """
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    frontier: list[int] = []
    best = -np.inf
    for i in order:
        if points[i][1] > best:
            frontier.append(i)
            best = points[i][1]
    return frontier


class Benchmark:
    """Runs every variant over one BEIR dataset with a given searcher."""

    def __init__(self, dataset: BeirDataset, searcher: HybridSearcher, settings: BenchSettings | None = None,
                 log: Log | None = None, checkpoint_dir: Path | None = None) -> None:
        self.dataset = dataset
        self.searcher = searcher
        self.settings = settings or BenchSettings()
        self.log = log or (lambda _msg: None)
        self.checkpoint_dir = checkpoint_dir
        qids = sorted(dataset.queries, key=lambda q: (len(q), q))
        if self.settings.max_queries is not None:
            qids = qids[: self.settings.max_queries]
        self.qids = qids
        self.qrels = {q: dataset.qrels[q] for q in qids}
        self.doc_ids = [d.id for d in searcher.index.docs]
        self.sparse: dict[str, Ranked] = {}
        self.dense: dict[str, Ranked] = {}
        self.sparse_ms: list[float] = []
        self.dense_ms: list[float] = []

    # ------------------------------------------------------------------ helpers
    def _evaluate(self, lists: dict[str, Ranked], k: int = 100) -> dict[str, float]:
        run = {q: [self.doc_ids[int(p)] for p in lists[q].positions[:k]] for q in self.qids}
        return {name: round(value, 4) for name, value in evaluate(run, self.qrels).items()}

    def _warm_up(self, fn: Callable[[str], object]) -> None:
        for qid in self.qids[: self.settings.warmup_queries]:
            fn(self.dataset.queries[qid])

    def retrieve(self) -> None:
        """First-stage retrieval for every query (shared by all fusion and rerank variants)."""
        k = self.settings.candidates
        self._warm_up(lambda q: (self.searcher.sparse(q, k), self.searcher.dense(q, k)))
        for qid in self.qids:
            query = self.dataset.queries[qid]
            self.sparse[qid], ms = self.searcher.sparse(query, k)
            self.sparse_ms.append(ms)
            self.dense[qid], ms = self.searcher.dense(query, k)
            self.dense_ms.append(ms)

    def fused(self, config: SearchConfig) -> tuple[dict[str, Ranked], list[float]]:
        """Fuse the cached first-stage lists for every query."""
        lists, times = {}, []
        for qid in self.qids:
            lists[qid], ms = self.searcher.fuse(self.sparse[qid], self.dense[qid], config)
            times.append(ms)
        return lists, times

    def rerank_scores(self, base: dict[str, Ranked], max_depth: int,
                      name: str = "base") -> dict[str, tuple[np.ndarray, list[float]]]:
        """Cross-encoder scores for the first ``max_depth`` hits of every query, scored in timed batches.

        Passages are scored ``rerank_step`` at a time and the cumulative wall-clock time is recorded after each
        batch, so ``cumulative[i]`` is the measured cost of reranking the first ``i * rerank_step`` hits. This is
        exactly the work a depth-``N`` rerank performs with a batch size of ``rerank_step``, so one pass yields a
        measured latency for every depth instead of re-scoring the same pairs once per depth.

        With a ``checkpoint_dir``, scores and their measured batch times are saved every 25 queries, so an
        interrupted run resumes where it stopped instead of re-scoring (the saved timings are the original
        measurements).
        """
        reranker = self.searcher.reranker
        if reranker is None:
            raise ValueError("the searcher has no reranker")
        step = self.settings.rerank_step
        docs = self.searcher.index.docs
        out = self._load_checkpoint(name, max_depth)
        if len(out) < len(self.qids):
            self._warm_up(lambda q: reranker.score(q, [docs[0].content] * step))
        for i, qid in enumerate(self.qids, start=1):
            if qid in out:
                continue
            query = self.dataset.queries[qid]
            passages = [docs[int(p)].content for p in base[qid].positions[:max_depth]]
            chunks, cumulative = [], [0.0]
            for start in range(0, len(passages), step):
                begin = time.perf_counter()
                chunks.append(reranker.score(query, passages[start : start + step]))
                cumulative.append(cumulative[-1] + (time.perf_counter() - begin) * 1000.0)
            out[qid] = (np.concatenate(chunks) if chunks else np.empty(0), cumulative)
            if i % 25 == 0 or i == len(self.qids):
                self._save_checkpoint(name, max_depth, out)
                self.log(f"    {i}/{len(self.qids)} queries scored")
        return out

    def _checkpoint_path(self, name: str, max_depth: int) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        slug = "".join(ch if ch.isalnum() else "-" for ch in name.lower())
        return self.checkpoint_dir / f"rerank_{slug}_top{max_depth}_step{self.settings.rerank_step}.json"

    def _save_checkpoint(self, name: str, max_depth: int, scored: dict[str, tuple[np.ndarray, list[float]]]) -> None:
        path = self._checkpoint_path(name, max_depth)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {qid: {"scores": scores.tolist(), "cumulative_ms": cum} for qid, (scores, cum) in scored.items()}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)

    def _load_checkpoint(self, name: str, max_depth: int) -> dict[str, tuple[np.ndarray, list[float]]]:
        path = self._checkpoint_path(name, max_depth)
        if path is None or not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded = {qid: (np.asarray(v["scores"], dtype=np.float64), [float(x) for x in v["cumulative_ms"]])
                  for qid, v in payload.items() if qid in self.qrels}
        self.log(f"    resumed {len(loaded)} scored queries from {path.name}")
        return loaded

    # ------------------------------------------------------------------ variants
    def first_stage_methods(self) -> list[MethodResult]:
        return [
            MethodResult("BM25", "bm25", {"k1": self.searcher.index.bm25.params.k1,
                                          "b": self.searcher.index.bm25.params.b},
                         self._evaluate(self.sparse), {"sparse": self.sparse_ms}),
            MethodResult("Dense", "dense", {}, self._evaluate(self.dense), {"dense": self.dense_ms}),
        ]

    def fusion_method(self, name: str, config: SearchConfig,
                      family: str = "hybrid") -> tuple[MethodResult, dict[str, Ranked]]:
        lists, fusion_ms = self.fused(config)
        params: dict[str, Any] = (
            {"fusion": "rrf", "rrf_k": config.rrf_k} if config.fusion == "rrf"
            else {"fusion": "convex", "alpha": config.alpha, "norm": config.norm})
        result = MethodResult(name, family, params, self._evaluate(lists),
                              {"sparse": self.sparse_ms, "dense": self.dense_ms, "fusion": fusion_ms})
        return result, lists

    def sweeps(self) -> dict[str, list[MethodResult]]:
        """RRF ``k`` and convex ``alpha`` (min-max and z-score) sweeps."""
        s = self.settings
        out: dict[str, list[MethodResult]] = {"rrf_k": [], "alpha_minmax": [], "alpha_zscore": []}
        for k in s.rrf_ks:
            cfg = SearchConfig(fusion="rrf", rrf_k=k)
            out["rrf_k"].append(self.fusion_method(f"RRF k={k:g}", cfg, "sweep")[0])
        for norm in ("minmax", "zscore"):
            for alpha in s.alphas:
                cfg = SearchConfig(fusion="convex", alpha=alpha, norm=norm)
                out[f"alpha_{norm}"].append(self.fusion_method(f"Convex {norm} a={alpha:g}", cfg, "sweep")[0])
        return out

    def rerank_sweep(self, base_name: str, base: dict[str, Ranked], base_stage_ms: dict[str, list[float]],
                     depths: Sequence[int]) -> list[MethodResult]:
        """Rerank the first ``N`` hits for every ``N`` in ``depths`` (multiples of ``rerank_step``)."""
        step = self.settings.rerank_step
        if any(d <= 0 or d % step for d in depths):
            raise ValueError(f"rerank depths must be positive multiples of rerank_step={step}")
        self.log(f"  cross-encoder scoring of the top {max(depths)} {base_name} hits")
        scored = self.rerank_scores(base, max(depths), base_name)
        results = []
        for depth in depths:
            lists, times = {}, []
            for qid in self.qids:
                scores, cumulative = scored[qid]
                n = min(depth, scores.size)
                start = time.perf_counter()
                lists[qid] = rerank_head(base[qid], n, scores[:n])
                sort_ms = (time.perf_counter() - start) * 1000.0
                times.append(cumulative[-(-n // step)] + sort_ms)
            results.append(MethodResult(f"{base_name} + rerank@{depth}", "rerank", {"depth": depth, "base": base_name},
                                        self._evaluate(lists), {**base_stage_ms, "rerank": times}))
        return results

    def filter_experiment(self) -> list[dict[str, Any]]:
        """Pre- vs post-filtering with a synthetic ``bucket`` field (uniform 0-99, seeded) on every document.

        A filter ``bucket < 100 * s`` keeps a share ``s`` of the corpus. Every query is run, so the page-fill
        figures (how often a full top-10 comes back) cover all queries. Quality is scored against judgements
        restricted to documents that pass the filter, over the queries that keep at least one relevant document;
        with very selective filters that subset is small, and its size is reported.
        """
        index = self.searcher.index
        rng = np.random.default_rng(self.settings.seed)
        buckets = rng.integers(0, 100, size=len(index))
        docs = [d.model_copy(update={"metadata": {"bucket": int(b)}})
                for d, b in zip(index.docs, buckets, strict=True)]
        searcher = HybridSearcher(HybridIndex(docs, index.bm25, index.dense, index.embedder_info),
                                  self.searcher.embedder, self.searcher.reranker)
        rows: list[dict[str, Any]] = []
        for share in self.settings.filter_selectivities:
            expr = Range(field="bucket", lt=100 * share)
            mask = searcher.index.metadata.mask(expr)
            allowed_ids = {self.doc_ids[i] for i in np.flatnonzero(mask)}
            qrels = {q: {d: g for d, g in rel.items() if d in allowed_ids} for q, rel in self.qrels.items()}
            qrels = {q: rel for q, rel in qrels.items() if rel}
            for mode in ("pre", "post"):
                cfg = SearchConfig(filter_mode=mode, candidates=self.settings.candidates, top_k=100)
                run, hits, totals = {}, [], []
                for qid in self.qids[: self.settings.warmup_queries]:  # warm-up, not timed
                    searcher.search(self.dataset.queries[qid], cfg, filter=expr, trace_k=0)
                for qid in self.qids:
                    resp = searcher.search(self.dataset.queries[qid], cfg, filter=expr, trace_k=0)
                    run[qid] = [h.doc_id for h in resp.hits]
                    hits.append(min(len(resp.hits), 10))
                    totals.append(resp.timings_ms["total"])
                metrics = {k: round(v, 4) for k, v in evaluate(run, qrels).items()} if qrels else None
                rows.append({"selectivity": share, "matching_docs": int(mask.sum()), "mode": mode,
                             "queries": len(self.qids), "judged_queries": len(qrels), "metrics": metrics,
                             "full_top10_rate": round(float(np.mean([h == 10 for h in hits])), 4),
                             "mean_hits_top10": round(float(np.mean(hits)), 2),
                             "latency_ms": latency_summary(totals)})
                quality = f"nDCG@10 {metrics['ndcg@10']:.3f}" if metrics else "no judged queries"
                self.log(f"  filter {share:.0%} {mode}: {quality}, full top-10 {rows[-1]['full_top10_rate']:.0%}")
        return rows


def bm25_without_stemming(index: HybridIndex) -> BM25Index:
    """Same corpus, same BM25 parameters, stemming off (an ablation)."""
    config = index.bm25.tokenizer.config.model_copy(update={"stemmer": "none"})
    return BM25Index.build([d.content for d in index.docs], config, index.bm25.params)


def environment() -> dict[str, Any]:
    """Interpreter, library and CPU details recorded next to the numbers."""
    import os

    import scipy

    info: dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                            "os": platform.system(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
                            "processor": platform.processor() or "unknown"}
    for dist in ("fastembed", "onnxruntime"):  # versions only; importing them would load the ONNX runtime
        try:
            info[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    return info


def run_sparse_benchmark(dataset: BeirDataset, *, k1_grid: Sequence[float] = (0.6, 0.9, 1.2, 1.5, 2.0),
                         b_grid: Sequence[float] = (0.2, 0.4, 0.6, 0.75, 0.9), candidates: int = 100,
                         log: Log | None = None) -> dict[str, Any]:
    """BM25-only benchmark: no models needed, so it runs anywhere in seconds.

    Reports BM25 with and without stemming at the default parameters (quality and per-query latency) and a
    ``k1`` x ``b`` sensitivity grid of nDCG@10 / Recall@100 on the test split. The grid is a sensitivity analysis:
    the default parameters are fixed in advance, not picked from it.
    """
    say = log or (lambda _msg: None)
    started = time.perf_counter()
    qids = sorted(dataset.queries, key=lambda q: (len(q), q))
    qrels = {q: dataset.qrels[q] for q in qids}
    doc_ids = [d.id for d in dataset.corpus]
    texts = [d.content for d in dataset.corpus]

    def run(index: BM25Index) -> tuple[dict[str, float], list[float]]:
        for qid in qids[:5]:  # warm-up
            index.search(dataset.queries[qid], candidates)
        lists, times = {}, []
        for qid in qids:
            start = time.perf_counter()
            ranked = index.search(dataset.queries[qid], candidates)
            times.append((time.perf_counter() - start) * 1000.0)
            lists[qid] = [doc_ids[int(p)] for p in ranked.positions]
        return {k: round(v, 4) for k, v in evaluate(lists, qrels).items()}, times

    methods = []
    build_s = {}
    variants: tuple[tuple[str, Literal["light", "none"]], ...] = (("BM25", "light"), ("BM25, no stemming", "none"))
    for label, stemmer in variants:
        start = time.perf_counter()
        index = BM25Index.build(texts, TokenizerConfig(stemmer=stemmer))
        build_s[label] = round(time.perf_counter() - start, 2)
        metrics, times = run(index)
        methods.append({"name": label, "family": "bm25", "params": {**index.params.model_dump(), "stemmer": stemmer},
                        "metrics": metrics, "latency_ms": {"sparse": latency_summary(times),
                                                           "total": latency_summary(times)},
                        "vocabulary": len(index.vocab)})
        say(f"  {label}: nDCG@10 {metrics['ndcg@10']:.3f}")
    index = BM25Index.build(texts, TokenizerConfig(stemmer="light"))
    grid = []
    for k1 in k1_grid:
        for b in b_grid:
            index.params = BM25Params(k1=k1, b=b)
            metrics, _ = run(index)
            grid.append({"k1": k1, "b": b, "metrics": metrics})
    say(f"  k1 x b grid: {len(grid)} settings")
    return {
        "dataset": dataset.name,
        "split": "test",
        "n_docs": len(dataset.corpus),
        "n_queries": len(qids),
        "environment": environment(),
        "methods": methods,
        "build_seconds": build_s,
        "grid": grid,
        "wall_time_s": round(time.perf_counter() - started, 1),
    }


def run_benchmark(dataset: BeirDataset, searcher: HybridSearcher, settings: BenchSettings | None = None, *,
                  int8: DenseIndex | None = None, ablation_searchers: dict[str, HybridSearcher] | None = None,
                  sweeps: bool = True, filters: bool = True, log: Log | None = None,
                  checkpoint_dir: Path | None = None) -> dict[str, Any]:
    """Run the full benchmark and return a JSON-ready result dictionary."""
    settings = settings or BenchSettings()
    say = log or (lambda _msg: None)
    bench = Benchmark(dataset, searcher, settings, say, checkpoint_dir)
    started = time.perf_counter()
    say(f"first-stage retrieval for {len(bench.qids)} queries")
    bench.retrieve()
    methods = bench.first_stage_methods()

    ablations: list[MethodResult] = []
    no_stem = bm25_without_stemming(searcher.index)
    lists, times = {}, []
    for qid in bench.qids:
        start = time.perf_counter()
        lists[qid] = no_stem.search(dataset.queries[qid], settings.candidates)
        times.append((time.perf_counter() - start) * 1000)
    ablations.append(MethodResult("BM25, no stemming", "ablation", {"stemmer": "none"}, bench._evaluate(lists),
                                  {"sparse": times}))
    for label, other in (ablation_searchers or {}).items():
        lists, times = {}, []
        for qid in bench.qids:
            lists[qid], ms = other.dense(dataset.queries[qid], settings.candidates)
            times.append(ms)
        ablations.append(MethodResult(label, "ablation", {}, bench._evaluate(lists), {"dense": times}))
    memory = None
    if int8 is not None and searcher.index.dense is not None:
        int8_searcher = HybridSearcher(HybridIndex(searcher.index.docs, searcher.index.bm25, int8,
                                                   searcher.index.embedder_info), searcher.embedder)
        lists, times = {}, []
        for qid in bench.qids:
            lists[qid], ms = int8_searcher.dense(dataset.queries[qid], settings.candidates)
            times.append(ms)
        methods.append(MethodResult("Dense int8", "dense", {"quantisation": "int8 per-dimension"},
                                    bench._evaluate(lists), {"dense": times}))
        memory = {"float32_bytes": searcher.index.dense.nbytes, "int8_bytes": int8.nbytes}

    rrf_result, rrf_lists = bench.fusion_method(f"Hybrid RRF (k={settings.rrf_k:g})",
                                                SearchConfig(fusion="rrf", rrf_k=settings.rrf_k))
    methods.append(rrf_result)
    for norm in ("minmax", "zscore"):
        cfg = SearchConfig(fusion="convex", alpha=settings.alpha, norm=norm)
        methods.append(bench.fusion_method(f"Hybrid convex (a={settings.alpha:g}, {norm})", cfg)[0])

    rerank: list[MethodResult] = []
    if searcher.reranker is not None and settings.rerank_depths:
        say("cross-encoder reranking")
        rerank = bench.rerank_sweep("Hybrid RRF", rrf_lists, rrf_result.stage_ms, settings.rerank_depths)
        headline = next((r for r in rerank if r.params["depth"] == settings.headline_depth), rerank[-1])
        methods.append(headline)
        if settings.bm25_rerank:
            methods.extend(bench.rerank_sweep("BM25", bench.sparse, {"sparse": bench.sparse_ms},
                                              [headline.params["depth"]]))

    sweep_results = bench.sweeps() if sweeps else {}
    say("sweeps done" if sweeps else "sweeps skipped")
    filter_rows = bench.filter_experiment() if filters else []

    headline_configs = {tuple(sorted(m.params.items())) for m in methods if m.family == "hybrid"}
    sweep_points = [r for rs in sweep_results.values() for r in rs
                    if tuple(sorted(r.params.items())) not in headline_configs]  # same config, already plotted
    unique: dict[str, MethodResult] = {r.name: r for r in methods + ablations + sweep_points + rerank}
    names = list(unique)
    points = [(latency_summary(unique[n].total_ms)["p95"], unique[n].metrics["ndcg@10"]) for n in names]
    frontier = [names[i] for i in pareto_frontier(points)]

    return {
        "dataset": dataset.name,
        "split": "test",
        "n_docs": len(searcher.index),
        "n_queries": len(bench.qids),
        "settings": settings.model_dump(),
        "environment": environment(),
        "models": {"embedder": searcher.embedder.name if searcher.embedder else None,
                   "reranker": searcher.reranker.name if searcher.reranker else None,
                   "tokenizer": searcher.index.bm25.tokenizer.config.model_dump(),
                   "bm25": searcher.index.bm25.params.model_dump()},
        "methods": [m.to_json() for m in methods],
        "ablations": [m.to_json() for m in ablations],
        "sweeps": {name: [r.to_json() for r in rs] for name, rs in sweep_results.items()},
        "rerank_sweep": [r.to_json() for r in rerank],
        "pareto_frontier": frontier,
        "all_points": [{"name": n, "family": unique[n].family, "ndcg@10": unique[n].metrics["ndcg@10"],
                        "p95_ms": round(latency_summary(unique[n].total_ms)["p95"], 3)} for n in names],
        "dense_memory": memory,
        "filtering": filter_rows,
        "wall_time_s": round(time.perf_counter() - started, 1),
    }

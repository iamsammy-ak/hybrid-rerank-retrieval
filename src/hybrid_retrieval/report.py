"""Render a benchmark result dictionary (from :func:`run_benchmark`) as Markdown."""

from __future__ import annotations

from typing import Any


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _ms(latency: dict[str, dict[str, float]], stage: str, key: str) -> str:
    return f"{latency[stage][key]:.1f}" if stage in latency else "-"


def _method_table(rows: list[dict[str, Any]]) -> list[str]:
    out = [_row(["Method", "nDCG@10", "Recall@100", "MRR@10", "p50 ms", "p95 ms"]),
           _row(["---", "---:", "---:", "---:", "---:", "---:"])]
    for m in rows:
        met, lat = m["metrics"], m["latency_ms"]
        out.append(_row([m["name"], f"{met['ndcg@10']:.3f}", f"{met['recall@100']:.3f}", f"{met['mrr@10']:.3f}",
                         _ms(lat, "total", "p50"), _ms(lat, "total", "p95")]))
    return out


def to_markdown(result: dict[str, Any], command: str) -> str:
    """A self-contained Markdown report with every table the README draws from."""
    env = result["environment"]
    lines = [
        f"# Benchmark: BEIR {result['dataset']} ({result['split']} split)",
        "",
        f"{result['n_docs']:,} documents, {result['n_queries']} judged queries. "
        f"Embedder `{result['models']['embedder']}`, reranker `{result['models']['reranker']}`. "
        f"BM25 k1={result['models']['bm25']['k1']}, b={result['models']['bm25']['b']}, "
        f"stemmer `{result['models']['tokenizer']['stemmer']}`.",
        "",
        f"Reproduce: `{command}`",
        "",
        f"Hardware: {env['cpu_count']} logical CPUs ({env['processor']}), {env['os']}, Python {env['python']}, "
        f"onnxruntime {env.get('onnxruntime', 'n/a')}. Latency is wall-clock per query, stages run sequentially "
        "in one process; p50/p95 are over queries.",
        "",
        "## Methods",
        "",
        *_method_table(result["methods"]),
        "",
        "## Per-stage latency (ms, p50 / p95)",
        "",
        _row(["Method", "Sparse", "Dense", "Fusion", "Rerank", "Total"]),
        _row(["---", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for m in result["methods"]:
        lat = m["latency_ms"]
        cells = [f"{_ms(lat, s, 'p50')} / {_ms(lat, s, 'p95')}" if s in lat else "-"
                 for s in ("sparse", "dense", "fusion", "rerank", "total")]
        lines.append(_row([m["name"], *cells]))
    if result.get("ablations"):
        lines += ["", "## Ablations", "", *_method_table(result["ablations"])]
    if result.get("dense_memory"):
        mem = result["dense_memory"]
        lines += ["", "## Dense index memory", "",
                  f"float32: {mem['float32_bytes'] / 1e6:.2f} MB, int8: {mem['int8_bytes'] / 1e6:.2f} MB "
                  f"({mem['float32_bytes'] / mem['int8_bytes']:.1f}x smaller)."]
    if result.get("rerank_sweep"):
        lines += ["", "## Rerank depth sweep", "", *_method_table(result["rerank_sweep"])]
    for name, title in (("rrf_k", "RRF k sweep"), ("alpha_minmax", "Convex alpha sweep (min-max)"),
                        ("alpha_zscore", "Convex alpha sweep (z-score)")):
        rows = result.get("sweeps", {}).get(name)
        if rows:
            lines += ["", f"## {title}", "", *_method_table(rows)]
    if result.get("filtering"):
        lines += ["", "## Pre- vs post-filtering (synthetic `bucket` metadata)", "",
                  "Each document gets a uniform random bucket 0-99 (seeded); the filter `bucket < 100*s` keeps a share "
                  "`s` of the corpus. All queries are run; quality uses judgements restricted to matching documents, "
                  "over the queries that keep a relevant document. Hybrid RRF, 100 candidates per retriever, no "
                  "rerank.", "",
                  _row(["Share kept", "Docs", "Mode", "Full top-10", "Mean hits in top 10", "Judged queries",
                        "nDCG@10", "Recall@100", "p95 ms"]),
                  _row(["---:", "---:", "---", "---:", "---:", "---:", "---:", "---:", "---:"])]
        for r in result["filtering"]:
            m = r["metrics"]
            lines.append(_row([f"{r['selectivity']:.0%}", str(r["matching_docs"]), r["mode"],
                               f"{r['full_top10_rate']:.0%}", f"{r['mean_hits_top10']:.1f}",
                               f"{r['judged_queries']} of {r['queries']}",
                               f"{m['ndcg@10']:.3f}" if m else "-", f"{m['recall@100']:.3f}" if m else "-",
                               f"{r['latency_ms']['p95']:.1f}"]))
    lines += ["", "## Pareto frontier (nDCG@10 vs p95 total latency)", "",
              ", ".join(result["pareto_frontier"]), "", f"Benchmark wall time: {result['wall_time_s']} s.", ""]
    return "\n".join(lines)


def sparse_to_markdown(result: dict[str, Any], command: str) -> str:
    """Markdown for :func:`run_sparse_benchmark` results."""
    env = result["environment"]
    lines = [
        f"# BM25 benchmark: BEIR {result['dataset']} ({result['split']} split)",
        "",
        f"{result['n_docs']:,} documents, {result['n_queries']} judged queries. No neural models involved.",
        "",
        f"Reproduce: `{command}`",
        "",
        f"Hardware: {env['cpu_count']} logical CPUs ({env['processor']}), {env['os']}, Python {env['python']}. "
        "Latency is wall-clock per query (tokenise, score, top-100), one query at a time.",
        "",
        "## Default parameters (k1 = 0.9, b = 0.4)",
        "",
        *_method_table(result["methods"]),
        "",
        "Index build time: " + ", ".join(f"{k} {v} s" for k, v in result["build_seconds"].items()) + ".",
        "",
        "## Sensitivity: nDCG@10 over k1 x b (light stemming)",
        "",
    ]
    b_values = sorted({row["b"] for row in result["grid"]})
    k1_values = sorted({row["k1"] for row in result["grid"]})
    cell = {(row["k1"], row["b"]): row["metrics"]["ndcg@10"] for row in result["grid"]}
    lines.append(_row(["k1 / b", *[f"{b:g}" for b in b_values]]))
    lines.append(_row(["---:"] * (len(b_values) + 1)))
    for k1 in k1_values:
        lines.append(_row([f"{k1:g}", *[f"{cell[(k1, b)]:.3f}" for b in b_values]]))
    lines += ["", f"Benchmark wall time: {result['wall_time_s']} s.", ""]
    return "\n".join(lines)

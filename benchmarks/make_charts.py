"""Draw the README charts from the benchmark results.

usage: python benchmarks/make_charts.py [scifact nfcorpus ...]
reads  benchmarks/results/bm25_<dataset>.json (from `bench --sparse-only`) and, when present,
       benchmarks/results/<dataset>.json (from the full `bench`)
writes assets/bm25_grid.png, assets/pareto_<dataset>.png, assets/methods_<dataset>.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAMILY_STYLE = {  # categorical slots 1-3 of the reference palette, plus a marker shape as secondary encoding
    "first stage": ("#2a78d6", "o"),
    "fusion": ("#eb6834", "s"),
    "rerank": ("#1baf7a", "D"),
}


def family(point: dict) -> str:
    if point["family"] in ("bm25", "dense", "ablation"):
        return "first stage"
    if point["family"] == "rerank":
        return "rerank"
    return "fusion"


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def pareto_chart(result: dict, out: Path) -> None:
    points = result["all_points"]
    frontier = set(result["pareto_frontier"])
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=110)
    fig.patch.set_facecolor("white")
    style_axes(ax)
    for name, (color, marker) in FAMILY_STYLE.items():
        sel = [p for p in points if family(p) == name]
        ax.scatter([p["p95_ms"] for p in sel], [p["ndcg@10"] for p in sel], s=34, color=color, marker=marker,
                   edgecolors="white", linewidths=1.2, label=name, zorder=3)
    front = sorted((p for p in points if p["name"] in frontier), key=lambda p: p["p95_ms"])
    xs = [p["p95_ms"] for p in front]
    ys = [p["ndcg@10"] for p in front]
    ax.step(xs, ys, where="post", color=INK, linewidth=1.6, zorder=2, label="Pareto frontier")
    ax.scatter(xs, ys, s=70, facecolors="none", edgecolors=INK, linewidths=1.4, zorder=4)
    labelled = {"BM25", "Dense", "Dense int8", f"Hybrid RRF (k={result['settings']['rrf_k']:g})"}
    labelled |= {p["name"] for p in points if p["family"] == "rerank"}
    for p in points:
        if p["name"] in labelled:
            text = p["name"].replace("Hybrid RRF + rerank@", "RRF + rerank@")
            ax.annotate(text, (p["p95_ms"], p["ndcg@10"]), xytext=(6, -11 if "BM25 +" in text else 5),
                        textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("p95 end-to-end latency per query (ms, log scale)", color=INK, fontsize=10)
    ax.set_ylabel("nDCG@10", color=INK, fontsize=10)
    ax.set_title(f"Quality vs latency on BEIR {result['dataset']} ({result['n_queries']} test queries)",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def methods_chart(result: dict, out: Path) -> None:
    methods = result["methods"]
    names = [m["name"] for m in methods][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(methods) + 1.4), dpi=110, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, metric, label in ((axes[0], "ndcg@10", "nDCG@10"), (axes[1], "recall@100", "Recall@100")):
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        values = [m["metrics"][metric] for m in methods][::-1]
        bars = ax.barh(names, values, color="#2a78d6", height=0.62, edgecolor="white", linewidth=2)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_width() + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center",
                    fontsize=8.5, color=INK)
        ax.set_xlim(0.0, min(1.05, max(values) + 0.12))
        ax.set_title(label, color=INK, fontsize=11, loc="left")
    axes[0].tick_params(axis="y", labelsize=9, colors=INK)
    fig.suptitle(f"BEIR {result['dataset']}: every method on the same {result['n_queries']} test queries",
                 color=INK_2, fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


SEQUENTIAL = LinearSegmentedColormap.from_list(  # one hue, light to dark (reference palette blue ramp)
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def bm25_grid_chart(results: list[dict], out: Path) -> None:
    """nDCG@10 over the k1 x b grid, one panel per dataset; the default setting is outlined."""
    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4.1), dpi=110, squeeze=False)
    fig.patch.set_facecolor("white")
    for ax, result in zip(axes[0], results, strict=True):
        k1s = sorted({row["k1"] for row in result["grid"]})
        bs = sorted({row["b"] for row in result["grid"]})
        cell = {(row["k1"], row["b"]): row["metrics"]["ndcg@10"] for row in result["grid"]}
        values = [[cell[(k1, b)] for b in bs] for k1 in k1s]
        lo, hi = min(map(min, values)), max(map(max, values))
        ax.imshow(values, cmap=SEQUENTIAL, vmin=lo, vmax=hi, aspect="auto", origin="lower")
        for i, k1 in enumerate(k1s):
            for j, b in enumerate(bs):
                v = cell[(k1, b)]
                dark = (v - lo) / ((hi - lo) or 1.0) > 0.4
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                        color="white" if dark else INK)
                if (k1, b) == (0.9, 0.4):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2))
        ax.set_xticks(range(len(bs)), [f"{b:g}" for b in bs])
        ax.set_yticks(range(len(k1s)), [f"{k1:g}" for k1 in k1s])
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel("b (length normalisation)", color=INK, fontsize=9.5)
        ax.set_ylabel("k1 (term-frequency saturation)", color=INK, fontsize=9.5)
        ax.set_title(f"BEIR {result['dataset']}: nDCG@10 (range {lo:.3f}-{hi:.3f})", color=INK, fontsize=10.5,
                     loc="left")
    fig.suptitle("BM25 sensitivity to k1 and b on the test split; outlined cell = the default (0.9, 0.4)",
                 color=INK_2, fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main() -> None:
    results_dir = ROOT / "benchmarks" / "results"
    datasets = sys.argv[1:] or ["scifact", "nfcorpus"]
    sparse = [json.loads((results_dir / f"bm25_{d}.json").read_text(encoding="utf-8"))
              for d in datasets if (results_dir / f"bm25_{d}.json").exists()]
    if sparse:
        bm25_grid_chart(sparse, ROOT / "assets" / "bm25_grid.png")
        print("wrote assets/bm25_grid.png")
    for dataset in datasets:
        full = results_dir / f"{dataset}.json"
        if not full.exists():
            continue
        result = json.loads(full.read_text(encoding="utf-8"))
        pareto_chart(result, ROOT / "assets" / f"pareto_{dataset}.png")
        methods_chart(result, ROOT / "assets" / f"methods_{dataset}.png")
        print(f"wrote assets/pareto_{dataset}.png and assets/methods_{dataset}.png")


if __name__ == "__main__":
    main()

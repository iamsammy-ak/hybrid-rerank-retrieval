"""``hybrid-search`` command line: download, index, search, bench, demo and serve."""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .beir import download_beir, load_beir
from .bm25 import BM25Params
from .demo import load_demo_corpus, load_models, run_demo
from .embedders import Embedder, FastEmbedEmbedder, load_embedder
from .filters import FilterError
from .index import HybridIndex, embedder_from_info
from .rerank import Reranker, load_reranker
from .schemas import Document
from .search import HybridSearcher, SearchConfig
from .tokenizer import TokenizerConfig

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Hybrid BM25 + dense retrieval with cross-encoder reranking and metadata filters.")
console = Console()


class EmbedderChoice(str, Enum):
    bge_small = "bge-small"
    hashing = "hashing"
    none = "none"


class RerankerChoice(str, Enum):
    auto = "auto"
    minilm = "minilm"
    overlap = "overlap"


class ModeChoice(str, Enum):
    bm25 = "bm25"
    dense = "dense"
    hybrid = "hybrid"


class FusionChoice(str, Enum):
    rrf = "rrf"
    convex = "convex"


class NormChoice(str, Enum):
    minmax = "minmax"
    zscore = "zscore"


class FilterModeChoice(str, Enum):
    pre = "pre"
    post = "post"


class StemmerChoice(str, Enum):
    light = "light"
    none = "none"


def _fail(message: str) -> NoReturn:
    console.print(f"[red]error:[/] {message}")
    raise typer.Exit(code=2)


def _progress_bar() -> Progress:
    return Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                    console=console, transient=True)


def _read_jsonl_docs(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as fh:
        return [Document.model_validate(json.loads(line)) for line in fh if line.strip()]


def _reranker_for(choice: RerankerChoice, embedder: Embedder | None, offline: bool) -> Reranker:
    if choice is RerankerChoice.auto:
        choice = RerankerChoice.minilm if isinstance(embedder, FastEmbedEmbedder) else RerankerChoice.overlap
    return load_reranker(choice.value, local_files_only=offline)


def _open_searcher(index_dir: Path | None, demo: bool, offline: bool, int8: bool,
                   reranker: RerankerChoice, need_reranker: bool) -> HybridSearcher:
    if demo:
        embedder, default_reranker, _ = load_models(offline)
        index = HybridIndex.build(load_demo_corpus(), embedder)
        chosen = default_reranker if reranker is RerankerChoice.auto else load_reranker(reranker.value)
        return HybridSearcher(index, embedder, chosen if need_reranker else None)
    if index_dir is None:
        _fail("pass --index DIR or --demo")
    loaded = HybridIndex.load(index_dir, quantised=int8)
    info = loaded.embedder_info
    query_embedder = embedder_from_info(info, local_files_only=offline) if info else None
    chosen_reranker = _reranker_for(reranker, query_embedder, offline) if need_reranker else None
    return HybridSearcher(loaded, query_embedder, chosen_reranker)


@app.command()
def download(names: list[str] = typer.Argument(..., help="BEIR dataset names, e.g. scifact nfcorpus"),
             data_dir: Path = typer.Option(Path("data/beir"), help="Where datasets are stored (git-ignored)."),
             force: bool = typer.Option(False, help="Re-download even if the folder exists.")) -> None:
    """Download BEIR datasets (official UKP archive, falling back to the pinned Hugging Face mirror)."""
    for name in names:
        folder, source = download_beir(name, data_dir, force=force)
        ds = load_beir(folder)
        console.print(f"[green]{name}[/]: {len(ds.corpus):,} documents, {len(ds.queries)} test queries "
                      f"(source: {source})")


@app.command()
def index(out: Path = typer.Option(..., help="Output index folder."),
          demo: bool = typer.Option(False, help="Index the bundled 48-document demo corpus."),
          beir: Path | None = typer.Option(None, help="A BEIR dataset folder (corpus.jsonl)."),
          jsonl: Path | None = typer.Option(None, help="JSONL with id, title, text, metadata per line."),
          embedder: EmbedderChoice = typer.Option(EmbedderChoice.bge_small, help="Dense embedder."),
          int8: bool = typer.Option(False, help="Also store int8-quantised embeddings."),
          k1: float = typer.Option(0.9, help="BM25 k1."), b: float = typer.Option(0.4, help="BM25 b."),
          stemmer: StemmerChoice = typer.Option(StemmerChoice.light, help="Light suffix stripping or none."),
          offline: bool = typer.Option(False, help="Never download models; use only the local cache.")) -> None:
    """Build and save an index (BM25 + embeddings + metadata)."""
    sources = [s for s in (demo, beir, jsonl) if s]
    if len(sources) != 1:
        _fail("pass exactly one of --demo, --beir, --jsonl")
    if demo:
        docs = load_demo_corpus()
    elif beir is not None:
        docs = load_beir(beir).corpus
    else:
        assert jsonl is not None
        docs = _read_jsonl_docs(jsonl)
    emb = None if embedder is EmbedderChoice.none else load_embedder(embedder.value, local_files_only=offline)
    with _progress_bar() as bar:
        task = bar.add_task(f"indexing {len(docs):,} documents", total=len(docs))
        built = HybridIndex.build(docs, emb, tokenizer_config=TokenizerConfig(stemmer=stemmer.value),
                                  bm25_params=BM25Params(k1=k1, b=b),
                                  progress=lambda done, _total: bar.update(task, completed=done))
        built.save(out, with_int8=int8)
    console.print(f"[green]saved[/] {len(docs):,} documents, vocabulary {len(built.bm25.vocab):,} terms, "
                  f"embedder {emb.name if emb else 'none'} -> {out.as_posix()}")


@app.command()
def search(query: str = typer.Argument(..., help="The query text."),
           index_dir: Path | None = typer.Option(None, "--index", help="Index folder from `index`."),
           demo: bool = typer.Option(False, help="Search the bundled demo corpus (indexed in memory)."),
           mode: ModeChoice = typer.Option(ModeChoice.hybrid, help="Which retrievers to run."),
           fusion: FusionChoice = typer.Option(FusionChoice.rrf, help="How hybrid mode combines the lists."),
           rrf_k: float = typer.Option(60.0, help="RRF constant k."),
           alpha: float = typer.Option(0.5, help="Convex weight on the dense list."),
           norm: NormChoice = typer.Option(NormChoice.minmax, help="Score normalisation for convex fusion."),
           candidates: int = typer.Option(100, help="Hits each retriever passes to fusion."),
           rerank: int = typer.Option(0, help="Rerank the top N fused hits (0 = off)."),
           reranker: RerankerChoice = typer.Option(RerankerChoice.auto, help="Which reranker to use."),
           top_k: int = typer.Option(10, help="Results to show."),
           filter: str | None = typer.Option(None, "--filter", help='e.g. \'category == "ml" and year >= 2022\''),
           filter_mode: FilterModeChoice = typer.Option(FilterModeChoice.pre,
                                                        help="pre (inside retrieval) or post (after fusion)."),
           int8: bool = typer.Option(False, help="Use the int8 embeddings."),
           offline: bool = typer.Option(False, help="Never download models; demo uses offline stand-ins."),
           as_json: bool = typer.Option(False, "--json", help="Print the full response as JSON.")) -> None:
    """Run one query and show the hits with per-stage timings."""
    try:
        config = SearchConfig(mode=mode.value, fusion=fusion.value, rrf_k=rrf_k, alpha=alpha, norm=norm.value,
                              candidates=candidates, rerank_depth=rerank, top_k=top_k, filter_mode=filter_mode.value)
    except ValueError as exc:
        _fail(str(exc))
    searcher = _open_searcher(index_dir, demo, offline, int8, reranker, need_reranker=rerank > 0)
    try:
        resp = searcher.search(query, config, filter=filter)
    except (FilterError, ValueError) as exc:
        _fail(str(exc))
    if as_json:
        typer.echo(resp.model_dump_json(indent=2))
        return
    title = f'"{query}"  [dim]{config.label}[/]'
    if resp.filter:
        title += f"\nfilter ({resp.filter_mode}): [magenta]{resp.filter}[/]  matches {resp.matching_docs} docs"
    table = Table(title=title, title_justify="left", header_style="bold cyan", expand=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("score", justify="right", width=8)
    table.add_column("doc", width=12, no_wrap=True)
    table.add_column("title", ratio=1, overflow="ellipsis", no_wrap=True)
    table.add_column("metadata", ratio=1, overflow="ellipsis", no_wrap=True)
    for hit in resp.hits:
        meta = ", ".join(f"{k}={v}" for k, v in hit.metadata.items())
        table.add_row(str(hit.rank), f"{hit.score:.3f}", hit.doc_id, hit.title, meta)
    console.print(table)
    console.print("[dim]" + " · ".join(f"{k} {v:.1f} ms" for k, v in resp.timings_ms.items()) + "[/]")


@app.command()
def bench(dataset: str = typer.Option("scifact", help="BEIR dataset name."),
          data_dir: Path = typer.Option(Path("data/beir"), help="Where datasets live."),
          index_dir: Path = typer.Option(Path("indexes"), help="Where built indexes are cached."),
          out: Path = typer.Option(Path("benchmarks/results"), help="Where JSON and Markdown results go."),
          rerank_depths: str = typer.Option("10,20,30,50", help="Comma-separated rerank depths, multiples of 10."),
          headline_depth: int = typer.Option(50, help="Rerank depth shown in the headline table."),
          max_queries: int | None = typer.Option(None, help="Evaluate only the first N queries."),
          sweeps: bool = typer.Option(True, help="Run the RRF k and convex alpha sweeps."),
          filters: bool = typer.Option(True, help="Run the pre- vs post-filter experiment."),
          bm25_rerank: bool = typer.Option(True, help="Also rerank the BM25 list at the headline depth."),
          sparse_only: bool = typer.Option(False, help="BM25 only (no models): defaults, no-stemming, k1 x b grid."),
          offline: bool = typer.Option(False, help="Never download models.")) -> None:
    """Benchmark every method on a BEIR dataset with the real models; writes JSON + Markdown."""
    from .evaluation import BenchSettings, run_benchmark, run_sparse_benchmark
    from .report import sparse_to_markdown, to_markdown

    folder, _ = download_beir(dataset, data_dir)
    ds = load_beir(folder)
    command = " ".join(["python -m hybrid_retrieval", *sys.argv[1:]]) if sys.argv[1:2] == ["bench"] else (
        f"python -m hybrid_retrieval bench --dataset {dataset}" + (" --sparse-only" if sparse_only else ""))
    out.mkdir(parents=True, exist_ok=True)
    if sparse_only:
        sparse = run_sparse_benchmark(ds, log=lambda m: console.print(f"[dim]{m}[/]"))
        sparse["command"] = command
        (out / f"bm25_{dataset}.json").write_text(json.dumps(sparse, indent=2), encoding="utf-8")
        (out / f"bm25_{dataset}.md").write_text(sparse_to_markdown(sparse, command), encoding="utf-8")
        console.print(f"[green]wrote[/] {(out / ('bm25_' + dataset)).as_posix()}.json and .md")
        return
    target = index_dir / dataset
    embedder = FastEmbedEmbedder(local_files_only=offline)
    if not (target / "manifest.json").exists():
        with _progress_bar() as bar:
            task = bar.add_task(f"embedding {len(ds.corpus):,} documents (one-off)", total=len(ds.corpus))
            HybridIndex.build(ds.corpus, embedder, progress=lambda done, _total: bar.update(task, completed=done)
                              ).save(target, with_int8=True)
    idx = HybridIndex.load(target)
    int8_index = HybridIndex.load(target, quantised=True).dense
    reranker = _reranker_for(RerankerChoice.minilm, embedder, offline)
    searcher = HybridSearcher(idx, embedder, reranker)
    no_instruction = embedder.with_prefixes(query_prefix="")  # shares the loaded ONNX session
    settings = BenchSettings(rerank_depths=[int(d) for d in rerank_depths.split(",") if d.strip()],
                             headline_depth=headline_depth, max_queries=max_queries, bm25_rerank=bm25_rerank)
    result = run_benchmark(ds, searcher, settings, int8=int8_index,
                           ablation_searchers={"Dense, no query instruction": HybridSearcher(idx, no_instruction)},
                           sweeps=sweeps, filters=filters, log=lambda m: console.print(f"[dim]{m}[/]"),
                           checkpoint_dir=target / "bench_checkpoints")
    result["command"] = command
    (out / f"{dataset}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / f"{dataset}.md").write_text(to_markdown(result, command), encoding="utf-8")
    for m in result["methods"]:
        console.print(f"{m['name']:<36} nDCG@10 {m['metrics']['ndcg@10']:.3f}  "
                      f"R@100 {m['metrics']['recall@100']:.3f}  p95 {m['latency_ms']['total']['p95']:.1f} ms")
    console.print(f"[green]wrote[/] {(out / dataset).as_posix()}.json and .md")


@app.command()
def demo(offline: bool = typer.Option(False, help="Force the offline stand-ins even if models are cached."),
         save_svg: Path | None = typer.Option(None, help="Also save the output as an SVG screenshot.")) -> None:
    """Walk through three queries on the bundled corpus, showing every stage's top hits."""
    out = Console(record=True, width=110) if save_svg else console
    run_demo(out, offline=offline)
    if save_svg:
        svg = out.export_svg(title="hybrid-search demo")
        lines = [line for line in svg.splitlines() if not line.strip().startswith("<!--")]  # drop comment lines
        save_svg.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"[green]saved[/] {save_svg.as_posix()}")


@app.command()
def serve(index_dir: Path | None = typer.Option(None, "--index", help="Index folder from `index`."),
          demo: bool = typer.Option(False, help="Serve the bundled demo corpus."),
          host: str = typer.Option("127.0.0.1", help="Bind address."),
          port: int = typer.Option(8702, help="Port."),
          offline: bool = typer.Option(False, help="Never download models.")) -> None:
    """Serve POST/GET /search over HTTP (FastAPI + uvicorn)."""
    import uvicorn

    from .api import create_app

    searcher = _open_searcher(index_dir, demo, offline, False, RerankerChoice.auto, need_reranker=True)
    uvicorn.run(create_app(searcher), host=host, port=port)


def main() -> None:
    """Console-script entry point."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower() not in ("utf-8", "utf8"):
            stream.reconfigure(encoding="utf-8", errors="replace")  # legacy Windows code pages lack the glyphs used
    app()


if __name__ == "__main__":
    main()

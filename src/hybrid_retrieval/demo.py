"""The bundled demo: a 48-document engineering knowledge base and three queries that show each stage at work.

The corpus (``data/demo_corpus.jsonl``) was written for this project. Every document carries ``year``,
``category`` and ``source`` metadata so filters can be demonstrated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .embedders import Embedder, FastEmbedEmbedder, HashingEmbedder
from .filters import parse_filter
from .index import HybridIndex
from .rerank import CrossEncoderReranker, Reranker, TermOverlapReranker
from .schemas import Document, SearchResponse
from .search import HybridSearcher, SearchConfig


def load_demo_corpus() -> list[Document]:
    """The 48 bundled documents."""
    text = resources.files("hybrid_retrieval").joinpath("data/demo_corpus.jsonl").read_text(encoding="utf-8")
    return [Document.model_validate(json.loads(line)) for line in text.splitlines() if line.strip()]


@dataclass(frozen=True)
class DemoQuery:
    """A demo query, the document a reader would want, and what it illustrates."""

    text: str
    target: str
    lesson: str


DEMO_QUERIES = (
    DemoQuery("my computer overheats and gets sluggish", "kb-ops-05",
              "No word in common with the answer, so BM25 finds nothing; the dense retriever matches the meaning."),
    DemoQuery("PSI above 0.2", "kb-ml-01",
              "An acronym and a threshold: BM25 matches them exactly, the embedding does not. Fusion keeps the "
              "answer near the top and the cross-encoder, reading query and passage together, puts it first."),
)
FILTER_QUERY = DemoQuery("speed up slow database queries", "kb-db-01",
                         "Metadata filter applied before (pre) or after (post) retrieval of 5 candidates each.")
FILTER_TEXT = 'category == "databases" and year >= 2021'


def load_models(offline: bool) -> tuple[Embedder, Reranker, str]:
    """Real models if they are already in the local cache (no download), otherwise the offline stand-ins."""
    if not offline:
        try:
            return (FastEmbedEmbedder(local_files_only=True), CrossEncoderReranker(local_files_only=True),
                    "bge-small-en-v1.5 embeddings + ms-marco-MiniLM-L-6-v2 cross-encoder")
        except Exception:  # noqa: BLE001 - any failure (not installed, not cached) means "use the stand-ins"
            pass
    return (HashingEmbedder(), TermOverlapReranker(),
            "offline stand-ins: hashing embedder + term-overlap reranker (not semantic models)")


def _cell(resp: SearchResponse, stage: str, rank: int, target: str, width: int) -> Text:
    hits = next((s.hits for s in resp.stages if s.name == stage), [])
    if not hits and rank == 1:
        return Text("(no matching terms)", style="dim")
    if rank > len(hits):
        return Text("-", style="dim")
    hit = hits[rank - 1]
    title = hit.title if len(hit.title) <= width else hit.title[: width - 1].rstrip() + "…"
    return Text(title, style="bold green" if hit.doc_id == target else "")


def _rank_of(resp: SearchResponse, stage: str, target: str) -> str:
    hits = next((s.hits for s in resp.stages if s.name == stage), [])
    return next((f"#{h.rank}" for h in hits if h.doc_id == target), "-")


def render_stages(console: Console, resp: SearchResponse, query: DemoQuery, show: int = 3) -> None:
    """Side-by-side top hits of each stage, with the wanted document highlighted."""
    names = [("sparse", "BM25"), ("dense", "Dense"), ("fusion", "RRF fusion"), ("rerank", "Reranked")]
    table = Table(title=f'[bold]"{query.text}"[/]', title_justify="left", expand=True, show_lines=False,
                  header_style="bold cyan", caption=query.lesson, caption_justify="left")
    table.add_column("#", width=2, justify="right")
    for stage, label in names:
        ms = resp.timings_ms.get(stage, 0.0)
        table.add_column(f"{label} [dim]{ms:.1f} ms[/]", ratio=1, no_wrap=True, overflow="ellipsis")
    for rank in range(1, show + 1):
        table.add_row(str(rank), *[_cell(resp, s, rank, query.target, 24) for s, _ in names])
    table.add_row("", *[Text(f"wanted doc: {_rank_of(resp, s, query.target)}", style="green") for s, _ in names])
    console.print(table)


def run_demo(console: Console, *, offline: bool = False) -> None:
    """Index the demo corpus in memory and walk through the demo queries."""
    embedder, reranker, label = load_models(offline)
    docs = load_demo_corpus()
    index = HybridIndex.build(docs, embedder)
    searcher = HybridSearcher(index, embedder, reranker)
    console.rule("[bold]Hybrid retrieval demo")
    console.print(f"{len(docs)} documents · {label}\n"
                  "Pipeline: BM25 + dense top-100 → reciprocal rank fusion (k=60) → rerank top 20\n")
    config = SearchConfig(mode="hybrid", fusion="rrf", rrf_k=60, candidates=100, rerank_depth=20, top_k=5)
    searcher.search("warm-up", config)  # first call pays one-off ONNX session start-up
    for query in DEMO_QUERIES:
        render_stages(console, searcher.search(query.text, config, trace_k=5), query)
        console.print()

    table = Table(title=f'[bold]"{FILTER_QUERY.text}"[/]  filter: [magenta]{FILTER_TEXT}[/]', title_justify="left",
                  expand=True, header_style="bold cyan", caption=FILTER_QUERY.lesson, caption_justify="left")
    table.add_column("mode", width=12)
    table.add_column("hits", width=4, justify="right")
    table.add_column("results (year, category)", ratio=1, no_wrap=True, overflow="ellipsis")
    for mode in ("pre", "post"):
        cfg = config.model_copy(update={"filter_mode": mode, "candidates": 5, "rerank_depth": 5})
        resp = searcher.search(FILTER_QUERY.text, cfg, filter=FILTER_TEXT)
        names = ", ".join(f"{h.doc_id} ({h.metadata.get('year')})" for h in resp.hits) or "(none)"
        table.add_row(f"{mode}-filter", str(len(resp.hits)), names)
    console.print(table)
    matching = int(index.metadata.mask(parse_filter(FILTER_TEXT)).sum())
    console.print(f"[dim]{matching} of {len(docs)} documents match the filter. Post-filtering drops non-matching "
                  "candidates after retrieval, so it can return fewer hits.[/]")


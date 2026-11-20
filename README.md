<p align="center">
  <img src="assets/banner.svg" alt="Hybrid Retrieval: sparse + dense search, fused and reranked, with measured trade-offs" width="100%">
</p>

<p align="center">
  <a href="https://github.com/iamsammy-ak/hybrid-rerank-retrieval/actions/workflows/ci.yml"><img src="https://github.com/iamsammy-ak/hybrid-rerank-retrieval/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/data-BEIR%20SciFact%20%C2%B7%20NFCorpus-4F46E5" alt="BEIR SciFact and NFCorpus">
  <img src="https://img.shields.io/badge/runs%20on-CPU%20%C2%B7%20ONNX-0F766E" alt="Runs on CPU with ONNX">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22C55E" alt="MIT license"></a>
</p>

**Hybrid Rerank Retrieval** is a search stack that finds the right document when the query and the answer share
no words, and also when the query is nothing but an error code. It runs a from-scratch BM25 index and a dense
embedding index side by side, fuses their rankings, lets a cross-encoder reorder the head of the list, and
applies typed metadata filters. Every stage is timed, and a benchmark harness for BEIR datasets reports what each
stage buys in quality and costs in latency. On BEIR SciFact, fusion lifts nDCG@10 from 0.677 (BM25) and 0.713
(dense) to 0.733 at 19 ms per query, while an off-the-shelf cross-encoder costs seconds per query and lowers it.

![hybrid-search demo: each stage's top hits for three queries, with timings](assets/demo.svg)

<sub>`python -m hybrid_retrieval demo` with the real models. The highlighted title is the document a reader
would want; the bottom row shows where each stage ranked it. Stage timings were captured on a laptop that was
short of memory at the time, so read them as relative, not absolute.</sub>

## Why this exists

Keyword search (BM25) is precise on exact terms such as product codes, acronyms and numbers, but it misses a
paraphrase: "my computer overheats" never matches "the processor runs too hot". Dense embeddings handle the
paraphrase and blur the exact term. Retrieval-augmented generation (RAG) and support search need both, plus a
reranker to put the best answer first, plus filters such as "only runbooks from 2023 onwards", all inside a
latency budget.

Each of those choices is a trade-off, and the usual advice ("just add a reranker") skips the numbers. This
repository builds every piece small enough to read, with a harness that measures how much each one adds and
what it costs.

## Features

| | |
|---|---|
| 🔤 **BM25 from scratch** | Okapi BM25 on SciPy sparse matrices with a documented tokeniser and light stemming; matches a loop-based reference implementation to 1e-5 |
| 🧭 **Dense retrieval** | `BAAI/bge-small-en-v1.5` on ONNX Runtime, query instruction prefix, exact cosine top-k, optional int8 index that is 4x smaller |
| 🔀 **Rank fusion** | Reciprocal Rank Fusion (RRF) and a convex combination with min-max or z-score normalisation |
| 🎯 **Cross-encoder rerank** | `ms-marco-MiniLM-L-6-v2` rescores the top N fused hits in batches |
| 🏷️ **Typed metadata filters** | `eq`, `in`, `range`, `and`, `or`, `not` as JSON or text (`year >= 2021 and source in [docs, runbook]`), compiled to boolean masks, applied before or after retrieval |
| 📏 **Evaluation harness** | nDCG@10, Recall@100, MRR@10 and p50/p95 latency per stage, parameter sweeps and a quality-vs-latency Pareto frontier |
| 🖥️ **CLI and HTTP API** | `hybrid-search index · search · bench · demo · serve`, plus a FastAPI `/search` endpoint on port 8702 |

## How it works

```mermaid
flowchart LR
    Q["Query + optional filter"] --> M["filters.py<br/>filter → boolean mask"]
    M --> S["BM25 (sparse)<br/>top 100"]
    M --> D["bge-small (dense)<br/>top 100"]
    S --> F["Fusion<br/>RRF or convex"]
    D --> F
    F --> R["Cross-encoder<br/>rerank top N"]
    R --> H["Top-k hits<br/>+ per-stage trace"]
```

1. **Filter** (`filters.py`). A filter written as text or JSON becomes a typed expression tree, then a boolean
   mask over the corpus. With *pre-filtering* (the default) the mask is applied inside each retriever before its
   top-k is taken; with *post-filtering* non-matching hits are dropped after fusion.
2. **Sparse retrieval** (`bm25.py`). Text is normalised, split, stripped of stopwords and lightly stemmed. BM25
   weights for every (document, term) pair are precomputed into a sparse matrix whose columns are posting lists, so
   a query is one sparse matrix-vector product.
3. **Dense retrieval** (`embedders.py`, `dense.py`). The query gets the BGE instruction prefix and is embedded
   with `bge-small-en-v1.5` (384 dimensions); cosine similarity against every document is a single matrix-vector
   product, followed by an `argpartition` top-k. Exact search, no approximate index.
4. **Fusion** (`fusion.py`). RRF scores a document by `Σ 1 / (k + rank)` across the two lists. The convex
   combination normalises each list's scores (min-max or z-score) and blends them with a weight `alpha`.
5. **Rerank** (`rerank.py`). A cross-encoder reads query and passage together and rescores the first N fused
   hits; the rest keep their fused order below them, so Recall@100 stays comparable.
6. **Trace** (`search.py`). The response carries the final hits plus each stage's top hits and wall-clock time.

## Benchmarks

### The full pipeline on BEIR SciFact (measured)

5,183 abstracts, 300 test queries, 100 candidates per retriever. Every method runs on the same queries in one
process; latency is wall-clock per query, with each stage timed separately (p50 in the stage columns).

| Method | nDCG@10 | Recall@100 | MRR@10 | Sparse ms | Dense ms | Fusion ms | Rerank ms | Total p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.677 | 0.912 | 0.647 | 0.7 | | | | 0.7 / 0.9 |
| Dense (bge-small, float32) | 0.713 | 0.942 | 0.682 | | 18.1 | | | 18.1 / 27.9 |
| Dense, int8 index | 0.715 | 0.942 | 0.685 | | 23.0 | | | 23.0 / 32.6 |
| Hybrid RRF (k = 60) | 0.718 | 0.968 | 0.686 | 0.7 | 18.1 | 0.2 | | 19.0 / 28.8 |
| Hybrid convex (α = 0.5, min-max) | **0.733** | **0.972** | **0.701** | 0.7 | 18.1 | 0.1 | | 19.0 / 28.8 |
| Hybrid convex (α = 0.5, z-score) | **0.734** | **0.972** | **0.701** | 0.7 | 18.1 | 0.2 | | 19.0 / 29.0 |
| Hybrid RRF + cross-encoder, top 10 | 0.706 | 0.968 | 0.672 | 0.7 | 18.1 | 0.2 | 1,523 | 1,544 / 1,610 |
| Hybrid RRF + cross-encoder, top 20 | 0.701 | 0.968 | 0.668 | 0.7 | 18.1 | 0.2 | 3,029 | 3,047 / 3,176 |
| Hybrid RRF + cross-encoder, top 50 | 0.694 | 0.968 | 0.661 | 0.7 | 18.1 | 0.2 | 7,538 | 7,560 / 7,858 |

![nDCG@10 against p95 latency for every configuration, with the Pareto frontier](assets/pareto_scifact.png)

![nDCG@10 and Recall@100 per method on SciFact](assets/methods_scifact.png)

What the numbers say:

- **Hybrid beats either retriever, almost for free.** The convex blend adds +0.020 nDCG@10 and +0.030
  Recall@100 over dense retrieval alone (+0.056 and +0.060 over BM25), for about one extra millisecond: BM25 and
  fusion together take under 1 ms, so hybrid latency is dense latency.
- **Where the time goes.** The dense stage is 18 ms at p50, most of it encoding the query with the ONNX model:
  the search itself is a single 5,183 x 384 matrix-vector product, and dropping the nine-word query instruction
  alone cuts the stage to 14.5 ms. BM25 takes 0.7 ms and fusion 0.2 ms.
- **This cross-encoder makes SciFact worse.** `ms-marco-MiniLM-L-6-v2` lowers nDCG@10 from 0.718 (the RRF list
  it reranks) to 0.706 at depth 10 and 0.694 at depth 50, while adding 1.5 to 7.5 seconds per query on four CPU
  threads (about 1.5 s per batch of ten abstracts, roughly 150 ms per query-abstract pair). It was trained on
  web search queries; SciFact queries are scientific claims and its passages are long abstracts cut at 512
  tokens. For context, the BEIR paper reports 0.688 for this reranker on top of BM25 on SciFact (cited, not
  measured here), below what bge-small reaches on its own. No rerank setting is on the Pareto frontier, so on
  this data the right rerank depth is zero; the harness is what makes that visible. A domain-matched or larger
  reranker is the next thing to try.
- **Fusion settings matter more than the method name.** RRF with the conventional k = 60 recovers most of the
  recall but little nDCG; smaller k does better here (k = 1: 0.734), and convex α between 0.5 and 0.7 forms a
  plateau (0.729-0.736). These sweeps run on the test split, so they show sensitivity, not tuned settings. The
  frontier points between 28.7 and 29.1 ms differ by less than measurement noise; only their quality differs.
- **int8 costs no quality.** The quantised index scores 0.715 against 0.713 and uses 2.0 MB instead of 8.0 MB,
  but is 5 ms slower at p50 because NumPy de-quantises each block before the product (there is no int8 kernel).
- **The BGE query instruction did not help here.** Without it dense retrieval scores 0.720 nDCG@10 and 0.953
  Recall@100 (with it: 0.713 and 0.942). The default follows the model card; this is the kind of choice to
  validate on held-out data per corpus.
- **Pre- vs post-filtering.** With a synthetic filter that keeps 2% of the corpus, pre-filtering still returns a
  full top-10 for 100% of queries; post-filtering does so for 1% (3.7 hits on average). At 50% and 10% both fill
  the page for 99-100% of queries. Only 6 and 31 queries keep a judged document at 2% and 10%, too few to
  compare the two modes' ranking quality, so the page-fill rate is the result to read. Full table in the report.

Reproduce (about 40 minutes on this laptop, almost all of it cross-encoder scoring, which is checkpointed every
25 queries so an interrupted run resumes):

```bash
python -m hybrid_retrieval download scifact
python -m hybrid_retrieval bench --dataset scifact --no-bm25-rerank --threads 4
python benchmarks/make_charts.py scifact
```

`--no-bm25-rerank` skips the BM25 + cross-encoder baseline and `--threads 4` caps ONNX Runtime, both to fit the
8 GB development laptop. Full report, including the RRF `k` and convex `alpha` sweeps, the ablations and the
filter table: [`benchmarks/results/scifact.md`](benchmarks/results/scifact.md).

### BM25 on BEIR (measured)

| Dataset | Method | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms | Published BM25 nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| SciFact (5,183 docs, 300 queries) | BM25, light stemming | **0.677** | **0.912** | 0.647 | 0.8 | 2.7 | 0.665 |
| | BM25, no stemming | 0.665 | 0.886 | 0.632 | 0.8 | 3.7 | |
| NFCorpus (3,633 docs, 323 queries) | BM25, light stemming | **0.326** | **0.245** | 0.525 | 0.6 | 2.6 | 0.325 |
| | BM25, no stemming | 0.313 | 0.234 | 0.516 | 0.5 | 0.8 | |

![BM25 nDCG@10 over a k1 x b grid on SciFact and NFCorpus](assets/bm25_grid.png)

What this shows:

- The from-scratch BM25 lands on the published numbers. The last column is the BM25 baseline reported in the
  BEIR paper (Thakur et al., 2021, Elasticsearch); it is cited, not measured here. Matching it is the check that
  the tokeniser, IDF and length normalisation are right.
- The light stemmer is worth about +0.012 nDCG@10 and +0.026 Recall@100 on SciFact (+0.013 and +0.011 on
  NFCorpus): conflating `increase`/`increased`/`increasing` matters on scientific text.
- BM25 is not where quality hides. Across 25 `(k1, b)` settings nDCG@10 stays within 0.667-0.684 on SciFact and
  0.319-0.330 on NFCorpus; the default (0.9, 0.4), fixed before the sweep, is within 0.007 of the best cell. The
  grid is a sensitivity check on the test split, not a tuning run.
- A query costs under a millisecond at the median (tokenise, sparse product, top-100) on 5,000 documents.

Reproduce (downloads the data on first use, needs no models, about 30 seconds per dataset):

```bash
python -m hybrid_retrieval download scifact nfcorpus
python -m hybrid_retrieval bench --dataset scifact --sparse-only
python -m hybrid_retrieval bench --dataset nfcorpus --sparse-only
python benchmarks/make_charts.py
```

Full reports with the whole grid: [`benchmarks/results/bm25_scifact.md`](benchmarks/results/bm25_scifact.md),
[`benchmarks/results/bm25_nfcorpus.md`](benchmarks/results/bm25_nfcorpus.md).

### Data and licences

- **SciFact**: Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020. Claims and annotations
  CC BY 4.0; abstracts ODC-By 1.0 (from S2ORC).
- **NFCorpus**: Boteva et al., *A Full-Text Learning to Rank Dataset for Medical Information Retrieval*, ECIR
  2016. Free for academic use; NutritionFacts.org content is subject to its own terms.
- **BEIR** packaging: Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information
  Retrieval Models*, NeurIPS 2021 Datasets and Benchmarks.
- `download` fetches the official BEIR archive from UKP Lab (TU Darmstadt). When that server is unreachable (it
  was during development) it falls back to the BeIR organisation's Hugging Face mirror, pinned to a fixed
  revision and verified against SHA-256 checksums. Nothing is committed; data lives in the git-ignored `data/`.
- Hardware for the numbers above: AMD Ryzen 5 5600H (6 cores, 12 threads), 8 GB RAM, Windows 11, Python 3.13,
  ONNX Runtime 1.32 capped at 4 threads for the full run, one query at a time. The bundled 48-document demo
  corpus was written for this project.

## Quick start

```bash
git clone https://github.com/iamsammy-ak/hybrid-rerank-retrieval.git
cd hybrid-rerank-retrieval
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[models,api,dev]"
python -m hybrid_retrieval demo  # uses the real models if cached, else offline stand-ins
```

The first run without a model cache downloads about 150 MB of ONNX models (`bge-small-en-v1.5` and
`ms-marco-MiniLM-L-6-v2`). `python -m hybrid_retrieval demo --offline` never downloads anything: it swaps in a
hashing embedder and a term-overlap reranker, which are clearly labelled stand-ins, not semantic models.

Search the bundled 48-document knowledge base, or index your own JSONL (`id`, `title`, `text`, `metadata`):

```bash
hybrid-search search "speed up slow database queries" --demo --rerank 20 \
    --filter 'category == "databases" and year >= 2021'
hybrid-search index --jsonl my_docs.jsonl --out indexes/my_docs --int8
hybrid-search search "how do I rotate a leaked key" --index indexes/my_docs --rerank 20 --top-k 5
hybrid-search serve --index indexes/my_docs          # http://127.0.0.1:8702/docs
```

```bash
curl -s localhost:8702/search -H 'content-type: application/json' \
  -d '{"query": "rotate a leaked key", "top_k": 3, "rerank_depth": 20, "filter": "year >= 2022"}'
```

No API key is needed: every model runs locally on the CPU. `.env.example` lists the two optional variables
(`FASTEMBED_CACHE_PATH` for the model cache, `HF_HUB_OFFLINE=1` to forbid downloads).

## Use it as a library

```python
from hybrid_retrieval import Document, HybridIndex, HybridSearcher, SearchConfig
from hybrid_retrieval.demo import load_models

docs = [
    Document(id="a", title="Rotating leaked API keys", text="Revoke the exposed key, then issue a new one.",
             metadata={"year": 2023, "team": "security"}),
    Document(id="b", title="Thermal throttling", text="A hot CPU lowers its clock speed to protect itself.",
             metadata={"year": 2020, "team": "platform"}),
    Document(id="c", title="Error E1047", text="The migration tool could not take its advisory lock.",
             metadata={"year": 2025, "team": "platform"}),
]

embedder, reranker, label = load_models(offline=False)  # real models if cached, else stand-ins
index = HybridIndex.build(docs, embedder)
searcher = HybridSearcher(index, embedder, reranker)

config = SearchConfig(mode="hybrid", fusion="rrf", rerank_depth=3, top_k=2)
response = searcher.search("my laptop overheats", config, filter='team == "platform" and year >= 2019')
for hit in response.hits:
    print(hit.rank, hit.doc_id, hit.title, round(hit.score, 3))
print({stage: round(ms, 1) for stage, ms in response.timings_ms.items()})
```

Output (the scores are raw cross-encoder logits; timings vary by machine):

```
1 b Thermal throttling -10.812
2 c Error E1047 -11.433
```

The same script is in [`examples/library_usage.py`](examples/library_usage.py). `HybridIndex.save(path)` and
`HybridIndex.load(path)` persist everything; loading refuses a query embedder that differs from the one that
built the index, because mixed embedding spaces fail silently.

## Design decisions

- **BM25 as sparse linear algebra, not a search library.** Pyserini needs a Java runtime and `rank_bm25` scores
  in Python loops. Precomputing every (document, term) BM25 weight into a CSC matrix turns a query into one
  sparse matrix-vector product that only touches the query terms' posting lists, persisted as two arrays and a
  vocabulary. Trade-off: changing `k1` or `b` re-derives the matrix (one vectorised pass), and there is no
  positional index, so no phrase queries.
- **Exact dense search instead of an approximate index.** For thousands to a few hundred thousand 384-dimension
  vectors, a brute-force matrix-vector product (about two million multiply-adds for 5,000 documents) is cheap
  and loses nothing to approximation, so there is no HNSW graph to tune. Memory is handled by the optional int8
  index instead. Past roughly a million vectors, an approximate index (HNSW or IVF) becomes the better trade.
- **RRF as the default fusion, convex combination as the tunable option.** RRF uses only ranks, so it needs no
  score calibration. It was chosen as the default before any measurement. On SciFact the convex blend at α = 0.5
  then beat RRF with k = 60 by 0.015 nDCG@10, and RRF with a small k matched it. Those sweeps ran on the test
  split, so the honest next step is to pick `k` or `alpha` on SciFact's train split, not to change the default
  from these numbers.
- **Rerank the head, keep the tail.** The cross-encoder costs one transformer pass per candidate, so latency grows
  linearly with the depth N. Reordering only the first N hits and keeping the rest below them makes N a pure
  latency knob and keeps Recall@100 comparable across depths. Reranking is off by default (`rerank_depth=0`),
  which the SciFact result supports: there, this reranker cost seconds and lowered quality.
- **Pre-filtering by default.** Applying the mask inside each retriever before its top-k means a selective filter
  still returns a full page when enough documents match. Post-filtering is easier to bolt onto an existing
  vector store, but it throws away retrieved candidates that fail the filter, so selective filters return short
  or empty pages: in the demo, 3 hits survive post-filtering where pre-filtering returns 5. The benchmark's
  filter experiment measures the same effect at 50%, 10% and 2% selectivity.
- **Offline stand-ins rather than mocks.** A hashing embedder and a term-overlap reranker implement the same
  interfaces as the real models, so CI exercises the identical pipeline without downloading anything; tests that
  need the real models are marked and run only where the model cache exists.

## Project structure

```
hybrid-rerank-retrieval/
├── src/hybrid_retrieval/
│   ├── tokenizer.py      normalisation, stopwords, light stemmer
│   ├── bm25.py           Okapi BM25 on SciPy sparse matrices, save/load
│   ├── embedders.py      bge-small via fastembed, hashing stand-in
│   ├── dense.py          exact cosine top-k, int8 quantisation
│   ├── fusion.py         RRF and convex combination
│   ├── rerank.py         cross-encoder and term-overlap stand-in
│   ├── filters.py        filter AST, text parser, mask compiler
│   ├── index.py          persisted hybrid index (manifest + arrays)
│   ├── search.py         the staged pipeline with per-stage timing
│   ├── metrics.py        nDCG, Recall, MRR, latency percentiles
│   ├── evaluation.py     benchmark harness, sweeps, Pareto frontier
│   ├── beir.py           BEIR download (checksummed) and loading
│   ├── api.py · cli.py   FastAPI service and Typer CLI
│   └── demo.py           demo walkthrough over data/demo_corpus.jsonl
├── benchmarks/           make_charts.py, results/ (JSON + Markdown)
├── examples/             library_usage.py
├── tests/                pytest + hypothesis, real-model tests marked `models`
└── assets/               banner, demo terminal SVG, charts
```

## Testing

```bash
ruff check . && mypy && pytest -q
```

The suite runs in well under a minute without any model download. Highlights:

- BM25 is checked against a loop-based textbook implementation on Hypothesis-generated corpora, query terms,
  `k1` and `b` values.
- Top-k selection is checked against a stable full sort; RRF and convex fusion against hand-computed scores and
  invariants (output is the union of the inputs, sorted).
- Filters: every operator, precedence, error positions, and a property test that rendering any generated
  expression tree to text and parsing it back yields the same mask; `not` is always the exact complement.
- Metrics against hand-computed nDCG values and bounds; the Pareto frontier against its definition.
- The CLI (`index`, `search`, `demo`) and the FastAPI app (through `TestClient`, no server started) end to end.
- Six tests marked `models` run the real ONNX models when they are cached locally and are skipped in CI.

## Limitations

- Single process, in memory, exact search. That is the right shape up to roughly a million documents; beyond
  that it needs an approximate index, sharding and incremental updates (adding documents currently means a
  rebuild).
- The full pipeline was benchmarked on SciFact only; NFCorpus has BM25 numbers only, and the BM25 +
  cross-encoder baseline was skipped to fit the development laptop's memory.
- Latency was measured on a laptop CPU with ONNX Runtime capped at four threads, so absolute milliseconds,
  especially the cross-encoder's, are higher than on a server or GPU. The relative cost of the stages is the
  useful signal. Queries run one at a time, with no batching across queries.
- Only one reranker was tried. Its result on SciFact says something about this model on this domain, not about
  cross-encoders in general.
- SciFact and NFCorpus are small and specialised (scientific and medical). Results can change on other domains,
  and parameter grids are computed on the test split, so they show sensitivity, not tuned values.
- The filter experiment in the harness uses synthetic random metadata, because the BEIR datasets have none.
- The tokeniser and light stemmer are English-only. The cross-encoder truncates passages at 512 tokens and was
  trained on MS MARCO web passages, not scientific abstracts.

## Roadmap

- [ ] An HNSW index behind the same interface, with a recall-vs-latency comparison against exact search
- [ ] Tune fusion weights and rerank depth on SciFact's train split and report the test score once
- [ ] Try a stronger or domain-matched reranker (for example `bge-reranker-base`) against the same frontier
- [ ] Incremental indexing with append-only segments and periodic merges

## License

[MIT](LICENSE) © 2026 Abhishek Kumar

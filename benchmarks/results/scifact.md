# Benchmark: BEIR scifact (test split)

5,183 documents, 300 judged queries. Embedder `fastembed:BAAI/bge-small-en-v1.5`, reranker `cross-encoder:Xenova/ms-marco-MiniLM-L-6-v2`. BM25 k1=0.9, b=0.4, stemmer `light`.

Reproduce: `python -m hybrid_retrieval bench --dataset scifact --no-bm25-rerank --threads 4`

Hardware: 12 logical CPUs (AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD), Windows, Python 3.13.2, onnxruntime 1.32.0. Latency is wall-clock per query, stages run sequentially in one process; p50/p95 are over queries.

## Methods

| Method | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.677 | 0.912 | 0.647 | 0.7 | 0.9 |
| Dense | 0.713 | 0.942 | 0.682 | 18.1 | 27.9 |
| Dense int8 | 0.715 | 0.942 | 0.685 | 23.0 | 32.6 |
| Hybrid RRF (k=60) | 0.718 | 0.968 | 0.686 | 19.0 | 28.8 |
| Hybrid convex (a=0.5, minmax) | 0.733 | 0.972 | 0.701 | 19.0 | 28.8 |
| Hybrid convex (a=0.5, zscore) | 0.734 | 0.972 | 0.701 | 19.0 | 29.0 |
| Hybrid RRF + rerank@50 | 0.694 | 0.968 | 0.661 | 7560.0 | 7857.6 |

## Per-stage latency (ms, p50 / p95)

| Method | Sparse | Dense | Fusion | Rerank | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.7 / 0.9 | - | - | - | 0.7 / 0.9 |
| Dense | - | 18.1 / 27.9 | - | - | 18.1 / 27.9 |
| Dense int8 | - | 23.0 / 32.6 | - | - | 23.0 / 32.6 |
| Hybrid RRF (k=60) | 0.7 / 0.9 | 18.1 / 27.9 | 0.2 / 0.2 | - | 19.0 / 28.8 |
| Hybrid convex (a=0.5, minmax) | 0.7 / 0.9 | 18.1 / 27.9 | 0.1 / 0.2 | - | 19.0 / 28.8 |
| Hybrid convex (a=0.5, zscore) | 0.7 / 0.9 | 18.1 / 27.9 | 0.2 / 0.4 | - | 19.0 / 29.0 |
| Hybrid RRF + rerank@50 | 0.7 / 0.9 | 18.1 / 27.9 | 0.2 / 0.2 | 7537.7 / 7838.9 | 7560.0 / 7857.6 |

## Ablations

| Method | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25, no stemming | 0.665 | 0.886 | 0.632 | 0.6 | 0.7 |
| Dense, no query instruction | 0.720 | 0.953 | 0.685 | 14.5 | 24.0 |

## Dense index memory

float32: 7.96 MB, int8: 1.99 MB (4.0x smaller).

## Rerank depth sweep

| Method | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid RRF + rerank@10 | 0.706 | 0.968 | 0.672 | 1544.1 | 1610.0 |
| Hybrid RRF + rerank@20 | 0.701 | 0.968 | 0.668 | 3047.2 | 3175.7 |
| Hybrid RRF + rerank@30 | 0.700 | 0.968 | 0.667 | 4546.1 | 4749.4 |
| Hybrid RRF + rerank@50 | 0.694 | 0.968 | 0.661 | 7560.0 | 7857.6 |

## RRF k sweep

| Method | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| RRF k=1 | 0.734 | 0.968 | 0.701 | 19.1 | 28.8 |
| RRF k=5 | 0.733 | 0.968 | 0.699 | 19.0 | 28.8 |
| RRF k=10 | 0.731 | 0.968 | 0.694 | 18.9 | 28.8 |
| RRF k=20 | 0.723 | 0.968 | 0.689 | 18.9 | 28.7 |
| RRF k=40 | 0.720 | 0.968 | 0.689 | 18.9 | 28.7 |
| RRF k=60 | 0.718 | 0.968 | 0.686 | 19.0 | 28.7 |
| RRF k=100 | 0.716 | 0.968 | 0.685 | 19.0 | 28.8 |
| RRF k=200 | 0.716 | 0.968 | 0.685 | 18.9 | 28.8 |

## Convex alpha sweep (min-max)

| Method | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Convex minmax a=0 | 0.677 | 0.912 | 0.647 | 19.0 | 28.7 |
| Convex minmax a=0.1 | 0.695 | 0.958 | 0.667 | 19.0 | 28.8 |
| Convex minmax a=0.2 | 0.707 | 0.958 | 0.676 | 19.0 | 28.9 |
| Convex minmax a=0.3 | 0.715 | 0.962 | 0.684 | 19.0 | 28.8 |
| Convex minmax a=0.4 | 0.725 | 0.968 | 0.694 | 19.0 | 28.8 |
| Convex minmax a=0.5 | 0.733 | 0.972 | 0.701 | 19.0 | 28.8 |
| Convex minmax a=0.6 | 0.732 | 0.972 | 0.700 | 19.0 | 28.8 |
| Convex minmax a=0.7 | 0.729 | 0.972 | 0.700 | 19.0 | 28.8 |
| Convex minmax a=0.8 | 0.717 | 0.972 | 0.686 | 19.0 | 28.7 |
| Convex minmax a=0.9 | 0.718 | 0.968 | 0.688 | 19.0 | 28.8 |
| Convex minmax a=1 | 0.713 | 0.942 | 0.682 | 19.0 | 28.8 |

## Convex alpha sweep (z-score)

| Method | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Convex zscore a=0 | 0.677 | 0.912 | 0.647 | 19.1 | 28.8 |
| Convex zscore a=0.1 | 0.694 | 0.958 | 0.667 | 19.1 | 29.0 |
| Convex zscore a=0.2 | 0.708 | 0.958 | 0.679 | 19.0 | 28.8 |
| Convex zscore a=0.3 | 0.717 | 0.962 | 0.686 | 19.1 | 29.0 |
| Convex zscore a=0.4 | 0.729 | 0.968 | 0.697 | 19.2 | 29.0 |
| Convex zscore a=0.5 | 0.734 | 0.972 | 0.701 | 19.0 | 28.9 |
| Convex zscore a=0.6 | 0.736 | 0.972 | 0.704 | 19.2 | 29.1 |
| Convex zscore a=0.7 | 0.733 | 0.972 | 0.704 | 19.2 | 29.0 |
| Convex zscore a=0.8 | 0.721 | 0.968 | 0.691 | 19.1 | 29.0 |
| Convex zscore a=0.9 | 0.716 | 0.968 | 0.686 | 19.0 | 28.9 |
| Convex zscore a=1 | 0.713 | 0.942 | 0.682 | 19.1 | 28.8 |

## Pre- vs post-filtering (synthetic `bucket` metadata)

Each document gets a uniform random bucket 0-99 (seeded); the filter `bucket < 100*s` keeps a share `s` of the corpus. All queries are run; quality uses judgements restricted to matching documents, over the queries that keep a relevant document. Hybrid RRF, 100 candidates per retriever, no rerank.

| Share kept | Docs | Mode | Full top-10 | Mean hits in top 10 | Judged queries | nDCG@10 | Recall@100 | p95 ms |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | 2604 | pre | 100% | 10.0 | 173 of 300 | 0.747 | 0.986 | 29.5 |
| 50% | 2604 | post | 100% | 10.0 | 173 of 300 | 0.753 | 0.986 | 27.6 |
| 10% | 536 | pre | 100% | 10.0 | 31 of 300 | 0.767 | 1.000 | 28.1 |
| 10% | 536 | post | 99% | 10.0 | 31 of 300 | 0.790 | 0.968 | 29.5 |
| 2% | 100 | pre | 100% | 10.0 | 6 of 300 | 0.798 | 1.000 | 27.9 |
| 2% | 100 | post | 1% | 3.7 | 6 of 300 | 0.816 | 1.000 | 27.9 |

## Pareto frontier (nDCG@10 vs p95 total latency)

BM25, no stemming, BM25, Dense, no query instruction, RRF k=40, RRF k=20, Convex minmax a=0.7, RRF k=5, RRF k=1, Convex zscore a=0.6

Benchmark wall time: 2278.3 s.

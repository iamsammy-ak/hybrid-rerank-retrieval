# BM25 benchmark: BEIR scifact (test split)

5,183 documents, 300 judged queries. No neural models involved.

Reproduce: `python -m hybrid_retrieval bench --dataset scifact --sparse-only`

Hardware: 12 logical CPUs (AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD), Windows, Python 3.13.2. Latency is wall-clock per query (tokenise, score, top-100), one query at a time.

## Default parameters (k1 = 0.9, b = 0.4)

| Method | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.677 | 0.912 | 0.647 | 0.8 | 2.7 |
| BM25, no stemming | 0.665 | 0.886 | 0.632 | 0.8 | 3.7 |

Index build time: BM25 6.3 s, BM25, no stemming 5.41 s.

## Sensitivity: nDCG@10 over k1 x b (light stemming)

| k1 / b | 0.2 | 0.4 | 0.6 | 0.75 | 0.9 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 | 0.670 | 0.678 | 0.679 | 0.680 | 0.680 |
| 0.9 | 0.675 | 0.677 | 0.679 | 0.683 | 0.681 |
| 1.2 | 0.678 | 0.674 | 0.675 | 0.679 | 0.679 |
| 1.5 | 0.673 | 0.671 | 0.675 | 0.682 | 0.681 |
| 2 | 0.667 | 0.671 | 0.675 | 0.682 | 0.684 |

Benchmark wall time: 32.6 s.

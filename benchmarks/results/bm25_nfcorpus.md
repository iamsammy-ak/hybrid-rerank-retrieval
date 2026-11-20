# BM25 benchmark: BEIR nfcorpus (test split)

3,633 documents, 323 judged queries. No neural models involved.

Reproduce: `python -m hybrid_retrieval bench --dataset nfcorpus --sparse-only`

Hardware: 12 logical CPUs (AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD), Windows, Python 3.13.2. Latency is wall-clock per query (tokenise, score, top-100), one query at a time.

## Default parameters (k1 = 0.9, b = 0.4)

| Method | nDCG@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.326 | 0.245 | 0.525 | 0.6 | 2.6 |
| BM25, no stemming | 0.313 | 0.234 | 0.516 | 0.5 | 0.8 |

Index build time: BM25 5.58 s, BM25, no stemming 4.13 s.

## Sensitivity: nDCG@10 over k1 x b (light stemming)

| k1 / b | 0.2 | 0.4 | 0.6 | 0.75 | 0.9 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6 | 0.322 | 0.322 | 0.321 | 0.321 | 0.319 |
| 0.9 | 0.326 | 0.326 | 0.324 | 0.325 | 0.323 |
| 1.2 | 0.328 | 0.328 | 0.328 | 0.329 | 0.328 |
| 1.5 | 0.330 | 0.330 | 0.329 | 0.329 | 0.328 |
| 2 | 0.329 | 0.330 | 0.329 | 0.330 | 0.329 |

Benchmark wall time: 26.3 s.

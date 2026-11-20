"""Index a few documents and run a filtered hybrid query from Python.

Uses the real models when they are already cached locally, otherwise the offline stand-ins.
run: python examples/library_usage.py
"""

from __future__ import annotations

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

embedder, reranker, label = load_models(offline=False)
index = HybridIndex.build(docs, embedder)
searcher = HybridSearcher(index, embedder, reranker)

config = SearchConfig(mode="hybrid", fusion="rrf", rerank_depth=3, top_k=2)
response = searcher.search("my laptop overheats", config, filter='team == "platform" and year >= 2019')

print(label)
for hit in response.hits:
    print(hit.rank, hit.doc_id, hit.title, round(hit.score, 3))
print({stage: round(ms, 1) for stage, ms in response.timings_ms.items()})

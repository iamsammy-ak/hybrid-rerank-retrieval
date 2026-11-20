from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from hypothesis.configuration import set_hypothesis_home_dir

from hybrid_retrieval.demo import load_demo_corpus
from hybrid_retrieval.embedders import HashingEmbedder
from hybrid_retrieval.index import HybridIndex
from hybrid_retrieval.rerank import TermOverlapReranker
from hybrid_retrieval.schemas import Document
from hybrid_retrieval.search import HybridSearcher

# Keep Hypothesis' example database and caches out of the working tree.
set_hypothesis_home_dir(Path(tempfile.gettempdir()) / "hybrid-retrieval-hypothesis")


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may reach a language-model API or download a model, whatever the environment says."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _models_cached() -> bool:
    try:
        from hybrid_retrieval.embedders import FastEmbedEmbedder
        from hybrid_retrieval.rerank import CrossEncoderReranker

        FastEmbedEmbedder(local_files_only=True)
        CrossEncoderReranker(local_files_only=True)
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not any("models" in item.keywords for item in items):
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if _models_cached():
        return
    skip = pytest.mark.skip(reason="fastembed models are not in the local cache")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def demo_docs() -> list[Document]:
    return load_demo_corpus()


@pytest.fixture(scope="session")
def demo_index(demo_docs: list[Document]) -> HybridIndex:
    return HybridIndex.build(demo_docs, HashingEmbedder())


@pytest.fixture(scope="session")
def searcher(demo_index: HybridIndex) -> HybridSearcher:
    return HybridSearcher(demo_index, HashingEmbedder(), TermOverlapReranker())


@pytest.fixture()
def tiny_beir(tmp_path: Path) -> Path:
    """A three-document dataset in BEIR layout."""
    folder = tmp_path / "tiny"
    (folder / "qrels").mkdir(parents=True)
    (folder / "corpus.jsonl").write_text(
        '{"_id": "d1", "title": "Cats", "text": "Cats purr and sleep.", "metadata": {}}\n'
        '{"_id": "d2", "title": "Dogs", "text": "Dogs bark at the postman.", "metadata": {"nested": {"x": 1}}}\n'
        '{"_id": "d3", "title": "", "text": "Birds sing at dawn."}\n', encoding="utf-8")
    (folder / "queries.jsonl").write_text(
        '{"_id": "q1", "text": "why do cats purr"}\n{"_id": "q2", "text": "barking dogs"}\n'
        '{"_id": "q3", "text": "unjudged query"}\n', encoding="utf-8")
    (folder / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t2\nq2\td3\t0\n",
                                               encoding="utf-8")
    return folder

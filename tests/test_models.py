"""Checks against the real ONNX models. Skipped unless both are already in the local fastembed cache."""

from __future__ import annotations

import numpy as np
import pytest
from rich.console import Console

from hybrid_retrieval.demo import run_demo
from hybrid_retrieval.embedders import FastEmbedEmbedder
from hybrid_retrieval.rerank import CrossEncoderReranker

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def bge() -> FastEmbedEmbedder:
    return FastEmbedEmbedder(local_files_only=True)


@pytest.fixture(scope="module")
def cross_encoder() -> CrossEncoderReranker:
    return CrossEncoderReranker(local_files_only=True)


def test_bge_vectors_are_384d_unit_length(bge: FastEmbedEmbedder) -> None:
    docs = bge.embed_documents(["short text", "a much longer passage " * 20, "third"])
    assert docs.shape == (3, 384) and docs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-5)
    assert bge.embed_query("question").shape == (384,)


def test_length_sorted_batching_keeps_input_order(bge: FastEmbedEmbedder) -> None:
    texts = ["x " * 200, "tiny", "medium length text here", "y " * 50]
    batched = bge.embed_documents(texts)
    one_by_one = np.vstack([bge.embed_documents([t]) for t in texts])
    np.testing.assert_allclose(batched, one_by_one, atol=1e-4)


def test_query_instruction_changes_the_vector(bge: FastEmbedEmbedder) -> None:
    plain = FastEmbedEmbedder(local_files_only=True, query_prefix="")
    assert float(bge.embed_query("heart disease") @ plain.embed_query("heart disease")) < 0.999


def test_dense_retrieval_matches_meaning(bge: FastEmbedEmbedder) -> None:
    docs = bge.embed_documents(["The processor lowers its clock speed when it runs too hot.",
                                "Rotate leaked credentials immediately."])
    q = bge.embed_query("my computer overheats and gets sluggish")
    assert docs[0] @ q > docs[1] @ q


def test_cross_encoder_prefers_the_relevant_passage(cross_encoder: CrossEncoderReranker) -> None:
    scores = cross_encoder.score("how do I fix a deadlock between transactions",
                                 ["Lock rows in a consistent order and keep transactions short to avoid deadlocks.",
                                  "Dark mode should not flash the wrong theme."])
    assert scores.shape == (2,) and scores[0] > scores[1]
    assert cross_encoder.score("q", []).shape == (0,)


def test_demo_runs_with_real_models() -> None:
    console = Console(record=True, width=110)
    run_demo(console)
    text = console.export_text()
    assert "bge-small-en-v1.5" in text and "Detecting data drift" in text

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from hybrid_retrieval.api import create_app
from hybrid_retrieval.cli import app
from hybrid_retrieval.search import HybridSearcher


@pytest.fixture(scope="module")
def client(searcher: HybridSearcher) -> TestClient:
    return TestClient(create_app(searcher))


def test_health_and_fields(client: TestClient) -> None:
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["documents"] == 48
    assert client.get("/fields").json() == {"category": "categorical", "source": "categorical", "year": "numeric"}


def test_post_search_with_text_filter(client: TestClient) -> None:
    body = {"query": "slow queries", "top_k": 3, "filter": 'category == "databases" and year >= 2021',
            "rerank_depth": 5}
    resp = client.post("/search", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["hits"]) == 3
    assert all(h["metadata"]["category"] == "databases" and h["metadata"]["year"] >= 2021 for h in data["hits"])
    assert data["filter"] == '(category == "databases" and year >= 2021)'
    assert [s["name"] for s in data["stages"]] == ["sparse", "dense", "fusion", "rerank"]


def test_post_search_with_json_filter(client: TestClient) -> None:
    body = {"query": "keys", "mode": "bm25", "filter": {"op": "in", "field": "source", "values": ["runbook"]}}
    data = client.post("/search", json=body).json()
    assert data["hits"] and all(h["metadata"]["source"] == "runbook" for h in data["hits"])


def test_get_search(client: TestClient) -> None:
    data = client.get("/search", params={"q": "E1047", "top_k": 2}).json()
    assert data["hits"][0]["doc_id"] == "kb-db-07" and len(data["hits"]) == 2


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"query": "x", "filter": "colour == red"}, 400),
        ({"query": "x", "filter": "year >="}, 400),
        ({"query": "", "top_k": 3}, 422),
        ({"query": "x", "alpha": 3}, 422),
        ({"query": "x", "filter": {"op": "range", "field": "year"}}, 422),
    ],
)
def test_bad_requests(client: TestClient, body: dict[str, object], status: int) -> None:
    assert client.post("/search", json=body).status_code == status


runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_cli_demo_offline() -> None:
    result = runner.invoke(app, ["demo", "--offline"])
    assert result.exit_code == 0, result.output
    assert "Hybrid retrieval demo" in result.output and "post-filter" in result.output


def test_cli_demo_saves_svg(tmp_path: Path) -> None:
    out = tmp_path / "demo.svg"
    result = runner.invoke(app, ["demo", "--offline", "--save-svg", str(out)])
    assert result.exit_code == 0, result.output
    svg = out.read_text(encoding="utf-8")
    assert svg.startswith("<svg") and "<!--" not in svg


def test_cli_index_then_search(tmp_path: Path) -> None:
    idx = tmp_path / "idx"
    result = runner.invoke(app, ["index", "--demo", "--embedder", "hashing", "--int8", "--out", str(idx)])
    assert result.exit_code == 0, result.output
    assert (idx / "dense_int8.npz").exists()
    result = runner.invoke(app, ["search", "leaked credentials", "--index", str(idx), "--rerank", "5",
                                 "--filter", "year >= 2020", "--json", "--offline"])
    assert result.exit_code == 0, result.output
    data = json.loads(ANSI.sub("", result.output))
    assert data["hits"][0]["doc_id"] == "kb-sec-01"
    assert all(h["metadata"]["year"] >= 2020 for h in data["hits"])


def test_cli_search_table_and_errors(tmp_path: Path) -> None:
    ok = runner.invoke(app, ["search", "ssh hardening", "--demo", "--offline", "--mode", "bm25", "--top-k", "3"])
    assert ok.exit_code == 0 and "kb-sec-04" in ok.output
    bad = runner.invoke(app, ["search", "x", "--demo", "--offline", "--filter", "year >>= 3"])
    assert bad.exit_code == 2 and "error" in bad.output
    none = runner.invoke(app, ["search", "x"])
    assert none.exit_code == 2
    both = runner.invoke(app, ["index", "--out", str(tmp_path / "x")])
    assert both.exit_code == 2


def test_cli_index_from_jsonl(tmp_path: Path) -> None:
    src = tmp_path / "docs.jsonl"
    src.write_text('{"id": "a", "title": "Alpha", "text": "first doc", "metadata": {"team": "x"}}\n'
                   '{"id": "b", "text": "second doc"}\n', encoding="utf-8")
    result = runner.invoke(app, ["index", "--jsonl", str(src), "--embedder", "none", "--out", str(tmp_path / "i")])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["search", "second", "--index", str(tmp_path / "i"), "--mode", "bm25", "--json"])
    assert json.loads(ANSI.sub("", result.output))["hits"][0]["doc_id"] == "b"

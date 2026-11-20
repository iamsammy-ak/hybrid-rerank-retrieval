"""Download and load datasets in the BEIR format (corpus.jsonl, queries.jsonl, qrels/<split>.tsv).

The primary source is the official BEIR archive mirror hosted by UKP Lab, TU Darmstadt. When that server is
unreachable the same files are fetched from the BEIR organisation's Hugging Face mirror, pinned to a fixed
revision and verified against SHA-256 checksums. Nothing is committed to the repository: the files land in a
local, git-ignored data directory.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .schemas import Document, MetaValue

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
HF_URL = "https://huggingface.co/datasets/BeIR/{repo}/resolve/{revision}/{file}"
_USER_AGENT = "hybrid-rerank-retrieval/0.1"


@dataclass(frozen=True)
class _MirrorFile:
    repo: str
    revision: str
    file: str
    sha256: str
    target: str


#: Pinned files on the BeIR Hugging Face mirror (the last revisions before the switch to Parquet).
HF_MIRROR: dict[str, tuple[_MirrorFile, ...]] = {
    "scifact": (
        _MirrorFile("scifact", "984eed826375f18d27936c4a32bf0f8491e3f414", "corpus.jsonl.gz",
                    "2f4a2a264485e5b5b95ae9cd5534817447757423991a4767d6da811539c0853e", "corpus.jsonl"),
        _MirrorFile("scifact", "984eed826375f18d27936c4a32bf0f8491e3f414", "queries.jsonl.gz",
                    "c29574b9cb8fdf043df8494f689de15f4604ac4b585eb34bda9b6a389aa6c4d3", "queries.jsonl"),
        _MirrorFile("scifact-qrels", "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029", "test.tsv",
                    "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c", "qrels/test.tsv"),
    ),
    "nfcorpus": (
        _MirrorFile("nfcorpus", "e28763e68d85db0fa71652ba3c4afabf8c5b3bb7", "corpus.jsonl.gz",
                    "ae4d6ab2ab88d880e7e180776c956a09c19c43a703457742cd850d2422035802", "corpus.jsonl"),
        _MirrorFile("nfcorpus", "e28763e68d85db0fa71652ba3c4afabf8c5b3bb7", "queries.jsonl.gz",
                    "baba3504a2736b0e32265962f04acc4fecfaf9649e0ea08e49ff441246f75364", "queries.jsonl"),
        _MirrorFile("nfcorpus-qrels", "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65", "test.tsv",
                    "f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a", "qrels/test.tsv"),
    ),
}


@dataclass
class BeirDataset:
    """A BEIR dataset split held in memory."""

    name: str
    corpus: list[Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    def __post_init__(self) -> None:
        self.queries = {qid: text for qid, text in self.queries.items() if qid in self.qrels}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response, dest.open("wb") as fh:  # noqa: S310
        shutil.copyfileobj(response, fh, length=1 << 20)


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract a zip archive, refusing members that would escape ``dest`` (zip-slip)."""
    dest = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"unsafe path in archive: {member.filename!r}")
        zf.extractall(dest)


def _download_ukp(name: str, data_dir: Path, timeout: float) -> Path:
    archive = data_dir / f"{name}.zip"
    try:
        _fetch(BEIR_URL.format(name=name), archive, timeout)
        _safe_extract(archive, data_dir)
    finally:
        archive.unlink(missing_ok=True)
    return data_dir / name


def _download_hf(name: str, data_dir: Path, timeout: float) -> Path:
    if name not in HF_MIRROR:
        raise ValueError(f"no pinned Hugging Face mirror for {name!r}; known: {sorted(HF_MIRROR)}")
    target = data_dir / name
    for item in HF_MIRROR[name]:
        out = target / item.target
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        try:
            _fetch(HF_URL.format(repo=item.repo, revision=item.revision, file=item.file), tmp, timeout)
            actual = _sha256(tmp)
            if actual != item.sha256:
                raise ValueError(f"checksum mismatch for {item.repo}/{item.file}: got {actual}")
            if item.file.endswith(".gz"):
                with gzip.open(tmp, "rb") as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                shutil.copyfile(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
    return target


def download_beir(name: str, data_dir: Path | str, *, force: bool = False, timeout: float = 60.0) -> tuple[Path, str]:
    """Download a BEIR dataset into ``data_dir/name``; returns the folder and the source that served it.

    Tries the official UKP archive first and falls back to the pinned, checksum-verified Hugging Face mirror
    when the archive server cannot be reached. An existing folder is reused unless ``force`` is set.
    """
    data_dir = Path(data_dir)
    target = data_dir / name
    if (target / "corpus.jsonl").exists() and not force:
        return target, "cache"
    if target.exists():
        shutil.rmtree(target)
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        folder, source = _download_ukp(name, data_dir, timeout), "ukp"
    except (urllib.error.URLError, TimeoutError, OSError, zipfile.BadZipFile):
        if target.exists():
            shutil.rmtree(target)
        folder, source = _download_hf(name, data_dir, timeout), "huggingface"
    if not (folder / "corpus.jsonl").exists() or not (folder / "qrels" / "test.tsv").exists():
        raise FileNotFoundError(f"{folder} is missing corpus.jsonl or qrels/test.tsv")
    return folder, source


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_corpus(folder: Path) -> list[Document]:
    """Read ``corpus.jsonl`` into documents (BEIR metadata is kept when it is flat)."""
    docs = []
    for row in _read_jsonl(folder / "corpus.jsonl"):
        meta = row.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        flat: dict[str, MetaValue] = {str(k): v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
        docs.append(Document(id=str(row["_id"]), title=str(row.get("title") or ""), text=str(row.get("text") or ""),
                             metadata=flat))
    return docs


def load_queries(folder: Path) -> dict[str, str]:
    """Read ``queries.jsonl`` into ``{query_id: text}``."""
    return {str(row["_id"]): str(row["text"]) for row in _read_jsonl(folder / "queries.jsonl")}


def load_qrels(folder: Path, split: str = "test") -> dict[str, dict[str, int]]:
    """Read ``qrels/<split>.tsv`` into ``{query_id: {doc_id: grade}}`` (grades > 0 only)."""
    qrels: dict[str, dict[str, int]] = {}
    with (folder / "qrels" / f"{split}.tsv").open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        if header[:2] != ["query-id", "corpus-id"]:
            raise ValueError(f"unexpected qrels header: {header}")
        for row in reader:
            qid, did, grade = row[0], row[1], int(row[2])
            if grade > 0:
                qrels.setdefault(qid, {})[did] = grade
    return qrels


def load_beir(folder: Path | str, split: str = "test") -> BeirDataset:
    """Load a BEIR dataset folder; queries without judgements in ``split`` are dropped."""
    folder = Path(folder)
    return BeirDataset(name=folder.name, corpus=load_corpus(folder), queries=load_queries(folder),
                       qrels=load_qrels(folder, split))

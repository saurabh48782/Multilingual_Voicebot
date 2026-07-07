"""Unit tests for the ingestor's index-teardown path (clear_index) and the
manifest/removal/doc_id correctness."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import src.rag.bm25_store as bm25_mod
import src.rag.ingestor as ing
import src.rag.store as store_mod
from src.rag.bm25_store import BM25Store
from src.rag.chunker import Chunk
from src.rag.embedder import EMBEDDING_DIM
from src.rag.store import FAISSStore


@pytest.fixture
def temp_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point the ingestor's artifact paths at a temp dir and create them."""
    paths = {
        "faiss_index_path": tmp_path / "faiss.index",
        "faiss_metadata_path": tmp_path / "metadata.parquet",
        "faiss_manifest_path": tmp_path / "manifest.json",
        "bm25_corpus_path": tmp_path / "bm25_corpus.pkl",
        "bm25_index_dir": tmp_path / "bm25",
    }
    for name, path in paths.items():
        monkeypatch.setattr(ing, name, path)
    return paths


def _create_all(paths: dict[str, Path]) -> None:
    paths["faiss_index_path"].write_bytes(b"idx")
    paths["faiss_metadata_path"].write_bytes(b"meta")
    paths["faiss_manifest_path"].write_text("{}")
    paths["bm25_corpus_path"].write_bytes(b"corpus")
    paths["bm25_index_dir"].mkdir()
    (paths["bm25_index_dir"] / "data.idx").write_bytes(b"bm25")


def test_clear_index_removes_all_artifacts(temp_artifacts: dict[str, Path]) -> None:
    _create_all(temp_artifacts)

    removed = ing.clear_index()

    assert set(removed) == set(temp_artifacts.values())
    assert not any(p.exists() for p in temp_artifacts.values())


def test_clear_index_resets_store_singletons(
    temp_artifacts: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _create_all(temp_artifacts)
    # Prime both module-level singletons; clear_index must drop them so the
    # next get_store()/get_bm25_store() reloads the now-empty index.
    monkeypatch.setattr(store_mod, "_store", object())
    monkeypatch.setattr(bm25_mod, "_store", object())

    ing.clear_index()

    assert store_mod._store is None
    assert bm25_mod._store is None


def test_clear_index_is_idempotent(temp_artifacts: dict[str, Path]) -> None:
    # Nothing on disk yet → no-op, no error, empty result.
    assert ing.clear_index() == []

    _create_all(temp_artifacts)
    assert len(ing.clear_index()) == 5
    assert ing.clear_index() == []


def test_clear_index_skips_missing_artifacts(temp_artifacts: dict[str, Path]) -> None:
    # Only a subset present → only those are reported removed.
    temp_artifacts["faiss_index_path"].write_bytes(b"idx")
    temp_artifacts["bm25_corpus_path"].write_bytes(b"corpus")

    removed = ing.clear_index()

    assert set(removed) == {
        temp_artifacts["faiss_index_path"],
        temp_artifacts["bm25_corpus_path"],
    }


# _ingest_file / ingest_corpus correctness fixes
@pytest.fixture
def real_stores(
    temp_artifacts: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> tuple[FAISSStore, BM25Store]:
    """Real FAISSStore/BM25Store backed by the temp artifact paths, wired up
    as what get_store()/get_bm25_store() return inside ingestor.py."""
    store = FAISSStore(
        index_path=temp_artifacts["faiss_index_path"],
        metadata_path=temp_artifacts["faiss_metadata_path"],
    )
    store.load()
    bm25 = BM25Store(
        index_dir=temp_artifacts["bm25_index_dir"],
        corpus_path=temp_artifacts["bm25_corpus_path"],
    )
    bm25.load()
    monkeypatch.setattr(ing, "get_store", lambda: store)
    monkeypatch.setattr(ing, "get_bm25_store", lambda: bm25)
    monkeypatch.setattr(
        ing,
        "embed_passages",
        lambda texts, **kw: np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32),
    )
    return store, bm25


def _fake_chunk_file_factory(
    call_count: dict[str, int],
) -> Callable[[Path], list[Chunk]]:
    """One chunk per file; text embeds a call counter so re-ingesting the same
    unchanged-looking file yields a different chunk_id, mimicking the
    non-deterministic chunk_id churn that LLM translation produces."""

    def _fake_chunk_file(path: Path) -> list[Chunk]:
        call_count[str(path)] = call_count.get(str(path), 0) + 1
        text = f"{path.name} content v{call_count[str(path)]}"
        return [
            Chunk(
                doc_id=path.name,
                chunk_index=0,
                text=text,
                text_en=text,
                source=str(path),
                page_num=-1,
            )
        ]

    return _fake_chunk_file


def test_manifest_persisted_only_after_index_save_succeeds(
    real_stores: tuple[FAISSStore, BM25Store],
    temp_artifacts: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash in store.save()/bm25.save() must not leave the manifest
    recording a hash for chunks that were never durably persisted - otherwise
    that file is silently skipped as 'unchanged' on every future run."""
    store, _bm25 = real_stores
    monkeypatch.setattr(ing, "chunk_file", _fake_chunk_file_factory({}))

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    doc = corpus_dir / "doc.txt"
    doc.write_text("original content")

    def _boom() -> None:
        raise RuntimeError("simulated crash writing FAISS index to disk")

    original_save = store.save
    monkeypatch.setattr(store, "save", _boom)

    with pytest.raises(RuntimeError):
        ing.ingest_corpus(corpus_dir, translate=False)

    # Manifest must not have been written at all - the crash happened before
    # persistence succeeded.
    assert not temp_artifacts["faiss_manifest_path"].exists()

    # Fix the crash and re-run: the file must be re-ingested, not skipped.
    monkeypatch.setattr(store, "save", original_save)
    summary = ing.ingest_corpus(corpus_dir, translate=False)
    assert summary["doc.txt"] == 1
    assert json.loads(temp_artifacts["faiss_manifest_path"].read_text())


def test_deleted_corpus_file_is_pruned_from_index_and_manifest(
    real_stores: tuple[FAISSStore, BM25Store],
    temp_artifacts: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, bm25 = real_stores
    monkeypatch.setattr(ing, "chunk_file", _fake_chunk_file_factory({}))

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    doc = corpus_dir / "gone.txt"
    doc.write_text("will be deleted")

    ing.ingest_corpus(corpus_dir, translate=False)
    assert store.total_chunks == 1
    assert bm25.total_chunks == 1

    doc.unlink()
    ing.ingest_corpus(corpus_dir, translate=False)

    assert store.total_chunks == 0
    assert bm25.total_chunks == 0
    manifest = json.loads(temp_artifacts["faiss_manifest_path"].read_text())
    assert manifest == {}


def test_unconditional_removal_prevents_duplicate_on_manifest_desync(
    real_stores: tuple[FAISSStore, BM25Store],
    temp_artifacts: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the manifest entry for a file is lost (e.g. a prior crash) while
    its chunks remain in the index, re-ingesting must replace those chunks
    rather than append duplicates alongside them - even though the new
    chunk_id differs (non-deterministic translation) so upsert-by-ID alone
    would not have deduped them."""
    store, bm25 = real_stores
    monkeypatch.setattr(ing, "chunk_file", _fake_chunk_file_factory({}))

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    doc = corpus_dir / "c.txt"
    doc.write_text("stable content")

    ing.ingest_corpus(corpus_dir, translate=False)
    assert store.total_chunks == 1

    # Simulate manifest desync: drop the manifest entry without touching the
    # index, as if a prior run persisted the index but crashed before saving
    # the manifest.
    manifest_path = temp_artifacts["faiss_manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    manifest.clear()
    manifest_path.write_text(json.dumps(manifest))

    ing.ingest_corpus(corpus_dir, translate=False)

    # Old chunk must have been replaced, not duplicated alongside the new one.
    assert store.total_chunks == 1
    assert bm25.total_chunks == 1

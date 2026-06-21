"""Unit tests for the ingestor's index-teardown path (clear_index)."""

from pathlib import Path

import pytest

import src.rag.bm25_store as bm25_mod
import src.rag.ingestor as ing
import src.rag.store as store_mod


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

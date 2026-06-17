"""Unit tests for the FAISS vector store."""

from pathlib import Path
import numpy as np
import pytest
from src.rag.embedder import EMBEDDING_DIM
from src.rag.store import FAISSStore


def _unit_vec(pos: int) -> np.ndarray:
    """A normalised one-hot vector; distinct positions are orthonormal."""
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    v[pos] = 1.0
    return v


def _query(pos: int) -> np.ndarray:
    return _unit_vec(pos).reshape(1, -1)


def _meta_row(
    cid_hex: str,
    cid_int: int,
    doc_id: str,
    *,
    chunk_index: int = 0,
    text: str = "text",
) -> dict:
    return {
        "chunk_id": cid_hex,
        "chunk_id_int": cid_int,
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "text_en": text,
        "source": f"{doc_id}.txt",
        "page_num": 1,
    }


def _store(tmp_path: Path) -> FAISSStore:
    s = FAISSStore(
        index_path=tmp_path / "faiss.index",
        metadata_path=tmp_path / "meta.parquet",
    )
    s.load()
    return s


def test_upsert_and_search_returns_nearest(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert(
        chunk_ids_int=[1, 2, 3],
        chunk_ids_hex=["a1", "a2", "a3"],
        vectors=np.vstack([_unit_vec(0), _unit_vec(1), _unit_vec(2)]),
        meta_rows=[
            _meta_row("a1", 1, "d1", text="pm kisan scheme"),
            _meta_row("a2", 2, "d2", text="old age pension"),
            _meta_row("a3", 3, "d3", text="rural housing"),
        ],
    )

    hits = s.search(_query(1), top_k=3)
    assert hits, "expected at least one hit"
    assert hits[0].chunk_id == "a2"
    assert hits[0].text_en == "old age pension"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_orders_by_cosine_descending(tmp_path: Path) -> None:
    s = _store(tmp_path)
    # Query equal to vec(0); a second vector partially overlapping (0 and 1).
    mixed = _unit_vec(0) + _unit_vec(1)
    mixed /= np.linalg.norm(mixed)
    s.upsert(
        chunk_ids_int=[1, 2],
        chunk_ids_hex=["a1", "a2"],
        vectors=np.vstack([mixed.astype(np.float32), _unit_vec(0)]),
        meta_rows=[_meta_row("a1", 1, "d1"), _meta_row("a2", 2, "d2")],
    )

    hits = s.search(_query(0), top_k=2)
    assert [h.chunk_id for h in hits] == ["a2", "a1"]  # exact match ranks first
    assert hits[0].score > hits[1].score


def test_upsert_replaces_existing_id(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert(
        [1],
        ["a1"],
        _unit_vec(0).reshape(1, -1),
        [_meta_row("a1", 1, "d1", text="original")],
    )
    # Re-upsert the same hex id with new content/vector.
    s.upsert(
        [1],
        ["a1"],
        _unit_vec(5).reshape(1, -1),
        [_meta_row("a1", 1, "d1", text="updated")],
    )

    assert s.total_chunks == 1
    hits = s.search(_query(5), top_k=1)
    assert hits[0].chunk_id == "a1"
    assert hits[0].text_en == "updated"


def test_upsert_empty_is_noop(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert([], [], np.empty((0, EMBEDDING_DIM), dtype=np.float32), [])
    assert s.total_chunks == 0


def test_remove_doc_returns_count_and_filters(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert(
        [1, 2, 3],
        ["a1", "a2", "a3"],
        np.vstack([_unit_vec(0), _unit_vec(1), _unit_vec(2)]),
        [
            _meta_row("a1", 1, "d1", chunk_index=0),
            _meta_row("a2", 2, "d1", chunk_index=1),
            _meta_row("a3", 3, "d2"),
        ],
    )

    removed = s.remove_doc("d1")
    assert removed == 2
    assert s.total_chunks == 1
    # Surviving chunk is from d2 only.
    hits = s.search(_query(2), top_k=5)
    assert [h.doc_id for h in hits] == ["d2"]


def test_remove_unknown_doc_returns_zero(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert([1], ["a1"], _unit_vec(0).reshape(1, -1), [_meta_row("a1", 1, "d1")])
    assert s.remove_doc("nope") == 0
    assert s.total_chunks == 1


def test_search_empty_index_returns_empty(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.search(_query(0), top_k=5) == []


def test_search_top_k_clamped_to_ntotal(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert(
        [1, 2],
        ["a1", "a2"],
        np.vstack([_unit_vec(0), _unit_vec(1)]),
        [_meta_row("a1", 1, "d1"), _meta_row("a2", 2, "d2")],
    )
    # Asking for more than exist must not raise or emit padded (-1) rows.
    hits = s.search(_query(0), top_k=10)
    assert len(hits) == 2
    assert all(h.chunk_id in {"a1", "a2"} for h in hits)


def test_persist_and_reload_roundtrip(tmp_path: Path) -> None:
    index_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "meta.parquet"

    a = FAISSStore(index_path=index_path, metadata_path=metadata_path)
    a.load()
    a.upsert(
        [1, 2],
        ["a1", "a2"],
        np.vstack([_unit_vec(0), _unit_vec(1)]),
        [_meta_row("a1", 1, "d1", text="alpha"), _meta_row("a2", 2, "d2", text="beta")],
    )
    a.save()

    b = FAISSStore(index_path=index_path, metadata_path=metadata_path)
    b.load()
    assert b.total_chunks == 2
    hits = b.search(_query(0), top_k=1)
    assert hits[0].chunk_id == "a1"
    assert hits[0].text_en == "alpha"

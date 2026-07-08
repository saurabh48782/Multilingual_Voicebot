"""Unit tests for the hybrid Retriever (BM25 + vector + RRF + confidence gate).

Reranker is disabled in these tests so behaviour stays deterministic; the
RRF-only branch is exercised, which is enough to test the dual-threshold
confidence gate.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.rag.retriever as retriever_mod
from src.rag.retriever import Retriever
from src.rag.store import SearchResult
from src.utils.config import cfg


def _make_doc(
    score: float,
    text: str = "some text",
    chunk_id: str = "abc123",
    doc_id: str = "doc1",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=0,
        text_en=text,
        source="test.pdf",
        page_num=0,
        score=score,
    )


@pytest.fixture()
def mock_store() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_bm25() -> MagicMock:
    m = MagicMock()
    m.search.return_value = []  # default: BM25 finds nothing
    return m


@pytest.fixture()
def retriever(mock_store: MagicMock, mock_bm25: MagicMock) -> Retriever:
    return Retriever(store=mock_store, bm25=mock_bm25)


@pytest.fixture(autouse=True)
def _disable_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass cross-encoder rerank so vector scores propagate through RRF only."""
    new_cfg = dict(cfg)
    rag = dict(new_cfg["rag"])
    retr = dict(rag.get("retrieval", {}))
    retr["use_reranker"] = False
    # Confidence gate is applied to normalised RRF scores in the no-rerank branch,
    # so use lenient thresholds keyed to RRF, not raw cosine.
    rag["retrieval"] = retr
    rag["retrieval_threshold"] = 0.5
    rag["retrieval_gap_threshold"] = 0.1
    new_cfg["rag"] = rag
    monkeypatch.setattr(retriever_mod, "cfg", new_cfg)


def test_gate_passes_when_top_clearly_above_rest(
    retriever: Retriever, mock_store: MagicMock, mock_bm25: MagicMock
) -> None:
    """Top doc is reinforced by BM25 → its fused RRF score doubles, gap > threshold."""
    from src.rag.bm25_store import BM25Hit

    top = _make_doc(0.80, chunk_id="aaaa000000000001", doc_id="d1")
    other = _make_doc(0.40, chunk_id="aaaa000000000002", doc_id="d2")
    mock_store.search.return_value = [top, other]
    mock_bm25.search.return_value = [
        BM25Hit(chunk_id_int=int("aaaa000000000001", 16) % (2**63), doc_id="d1", score=5.0),
    ]

    with patch.object(
        retriever_mod, "embed_query", return_value=np.zeros((1, 1024), dtype=np.float32)
    ):
        result = retriever.search("any query")

    assert result.passed is True
    assert result.top_score == pytest.approx(1.0)
    assert result.docs[0].doc_id == "d1"


def test_gate_fails_when_gap_too_small(retriever: Retriever, mock_store: MagicMock) -> None:
    docs = [
        _make_doc(0.80, chunk_id="aaaa000000000001"),
        _make_doc(0.79, chunk_id="aaaa000000000002"),
    ]
    mock_store.search.return_value = docs

    with patch.object(
        retriever_mod, "embed_query", return_value=np.zeros((1, 1024), dtype=np.float32)
    ):
        result = retriever.search("any query")

    # With only two docs from dense and BM25 empty, normalised RRF gives
    # top=1.0 and gap ≈ 1/(60+1) / 1.0 ≈ small → gate fails.
    assert result.passed is False


def test_empty_store_returns_failed_gate(retriever: Retriever, mock_store: MagicMock) -> None:
    mock_store.search.return_value = []

    with patch.object(
        retriever_mod, "embed_query", return_value=np.zeros((1, 1024), dtype=np.float32)
    ):
        result = retriever.search("query")

    assert result.passed is False
    assert result.top_score == 0.0
    assert result.docs == []


def test_rrf_fuses_dense_and_bm25(
    retriever: Retriever, mock_store: MagicMock, mock_bm25: MagicMock
) -> None:
    """Doc ranked highly by *both* dense and BM25 should beat doc ranked only by dense."""
    from src.rag.bm25_store import BM25Hit

    dense_top = _make_doc(0.9, chunk_id="aaaa000000000001", doc_id="d1")
    dense_only = _make_doc(0.85, chunk_id="aaaa000000000002", doc_id="d2")
    mock_store.search.return_value = [dense_only, dense_top]  # dense order: d2, d1

    # BM25 ranks d1 first
    mock_bm25.search.return_value = [
        BM25Hit(chunk_id_int=int("aaaa000000000001", 16) % (2**63), doc_id="d1", score=5.0),
    ]

    with patch.object(
        retriever_mod, "embed_query", return_value=np.zeros((1, 1024), dtype=np.float32)
    ):
        result = retriever.search("query")

    assert result.docs, "expected at least one doc"
    # d1 appears in both rankings → highest fused score
    assert result.docs[0].doc_id == "d1"

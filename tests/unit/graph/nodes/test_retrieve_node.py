from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.graph.nodes.retrieve import make_retrieve, route_after_retrieve
from src.rag.retriever import RetrievalResult
from src.rag.store import SearchResult
from tests.unit.graph.nodes.conftest import make_doc


def _node(retriever: MagicMock | None = None):
    deps = MagicMock()
    if retriever is not None:
        deps.retriever = retriever
    return make_retrieve(deps)


def _retriever_returning(
    docs: list[SearchResult],
    passed: bool = True,
    top_score: float = 0.9,
    gap: float = 0.1,
):
    r = MagicMock()
    r.search.return_value = RetrievalResult(
        docs=docs, top_score=top_score, gap=gap, passed=passed
    )
    return r


# empty / blank query short-circuit
@pytest.mark.parametrize(
    "state",
    [
        {"rewritten_query": "", "english_query": ""},
        {"rewritten_query": "   ", "english_query": ""},
        {"english_query": ""},
        {},
    ],
    ids=["both-empty", "whitespace-rewritten", "english-empty", "no-keys"],
)
def test_blank_query_skips_retriever_and_fails_gate(state: dict) -> None:
    retriever = MagicMock()
    result = _node(retriever)(state)
    retriever.search.assert_not_called()
    assert result["retrieval_passed"] is False
    assert result["retrieved_docs"] == []
    assert result["retrieval_confidence"] == 0.0


def test_blank_query_uses_existing_fallback_reason() -> None:
    state = {"rewritten_query": "", "fallback_reason": "stt_error"}
    result = _node()(state)
    assert result["fallback_reason"] == "stt_error"


def test_blank_query_defaults_fallback_reason_to_empty_query() -> None:
    result = _node()({"rewritten_query": ""})
    assert result["fallback_reason"] == "empty_query"


# query preference
def test_prefers_rewritten_query_over_english_query() -> None:
    retriever = _retriever_returning([make_doc()])
    _node(retriever)({"rewritten_query": "rewritten", "english_query": "original"})
    retriever.search.assert_called_once_with("rewritten")


def test_falls_back_to_english_query_when_no_rewritten() -> None:
    retriever = _retriever_returning([make_doc()])
    _node(retriever)({"english_query": "What is PM Kisan?"})
    retriever.search.assert_called_once_with("What is PM Kisan?")


# successful retrieval
@pytest.mark.parametrize(
    ("passed", "top_score", "gap"),
    [
        (True, 0.91, 0.13),
        (False, 0.50, 0.02),
    ],
    ids=["gate-passed", "gate-failed"],
)
def test_successful_retrieval_returns_all_fields(
    passed: bool, top_score: float, gap: float
) -> None:
    docs = [make_doc(score=top_score)]
    retriever = _retriever_returning(docs, passed=passed, top_score=top_score, gap=gap)
    result = _node(retriever)({"rewritten_query": "PM Kisan eligibility"})
    assert result["retrieved_docs"] == docs
    assert result["retrieval_confidence"] == top_score
    assert result["retrieval_gap"] == gap
    assert result["retrieval_passed"] is passed
    assert "fallback_reason" not in result


# retriever exception
@pytest.mark.parametrize(
    "exc",
    [RuntimeError("index not loaded"), OSError("disk error"), Exception("boom")],
)
def test_retriever_exception_returns_retrieval_error_fallback(exc: Exception) -> None:
    retriever = MagicMock()
    retriever.search.side_effect = exc
    result = _node(retriever)({"rewritten_query": "PM Kisan?"})
    assert result == {
        "retrieved_docs": [],
        "retrieval_confidence": 0.0,
        "retrieval_gap": 0.0,
        "retrieval_passed": False,
        "fallback_reason": "retrieval_error",
    }


# confidence-gate router
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"retrieval_passed": True}, "generate"),
        ({"retrieval_passed": False}, "fallback"),
        ({}, "fallback"),
        ({"retrieval_passed": None}, "fallback"),
    ],
    ids=["passed-true", "passed-false", "key-absent", "passed-none"],
)
def test_route_after_retrieve(state: dict, expected: str) -> None:
    assert route_after_retrieve(state) == expected  # type: ignore[arg-type]

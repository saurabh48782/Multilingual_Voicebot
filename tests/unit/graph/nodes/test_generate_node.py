from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.graph.nodes.generate import _build_context, _strip_thinking, make_generate
from src.graph.prompts import INSUFFICIENT_CONTEXT
from src.rag.store import SearchResult


def _doc(text: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id="c1",
        doc_id="doc1",
        chunk_index=0,
        text_en=text,
        source="doc.pdf",
        page_num=1,
        score=score,
    )


def _node(llm_content: str | None = None, side_effect: Exception | None = None):
    deps = MagicMock()
    resp = MagicMock()
    if side_effect:
        deps.llm.complete.side_effect = side_effect
    else:
        resp.content = llm_content or ""
        deps.llm.complete.return_value = resp
    return make_generate(deps), deps


# no-docs short-circuit
@pytest.mark.parametrize(
    "state",
    [
        {"retrieved_docs": [], "rewritten_query": "What is PM Kisan?"},
        {"retrieved_docs": None, "rewritten_query": "What is PM Kisan?"},
        {"rewritten_query": "What is PM Kisan?"},
    ],
    ids=["empty-docs", "none-docs", "missing-docs-key"],
)
def test_no_docs_returns_fallback(state: dict) -> None:
    node, deps = _node()
    result = node(state)  # type: ignore[arg-type]
    deps.llm.complete.assert_not_called()
    assert result == {
        "english_response": "",
        "fallback_triggered": True,
        "fallback_reason": "no_context",
    }


# query preference: rewritten_query > english_query
@pytest.mark.parametrize(
    ("state", "expected_query"),
    [
        (
            {
                "retrieved_docs": [_doc("PM Kisan: ₹6000/year direct income support.")],
                "rewritten_query": "What is PM Kisan about?",
                "english_query": "What is it?",
            },
            "What is PM Kisan about?",
        ),
        (
            {
                "retrieved_docs": [_doc("PM Kisan: ₹6000/year.")],
                "english_query": "What is PM Kisan?",
            },
            "What is PM Kisan?",
        ),
    ],
    ids=["rewritten-query-preferred", "falls-back-to-english-query"],
)
def test_query_preference(state: dict, expected_query: str) -> None:
    node, deps = _node("PM Kisan gives ₹6000 per year.")
    node(state)  # type: ignore[arg-type]
    call_kwargs = deps.llm.complete.call_args
    assert expected_query in call_kwargs.kwargs["messages"][-1]["content"]


# successful generation
def test_successful_generation_returns_english_response() -> None:
    answer = "PM Kisan provides ₹6000 per year."
    node, _ = _node(answer)
    state = {
        "retrieved_docs": [_doc("PM Kisan: ₹6000/year direct income support.")],
        "rewritten_query": "What is PM Kisan?",
    }
    result = node(state)  # type: ignore[arg-type]
    assert result == {"english_response": answer}


# insufficient-context sentinel
@pytest.mark.parametrize(
    "llm_answer",
    [
        INSUFFICIENT_CONTEXT,
        INSUFFICIENT_CONTEXT + " I cannot find relevant information.",
    ],
    ids=["exact-sentinel", "sentinel-with-trailing-text"],
)
def test_insufficient_context_sentinel_triggers_fallback(llm_answer: str) -> None:
    node, _ = _node(llm_answer)
    state = {
        "retrieved_docs": [_doc("Some context.")],
        "rewritten_query": "Unrelated query",
    }
    result = node(state)  # type: ignore[arg-type]
    assert result == {
        "english_response": "",
        "fallback_triggered": True,
        "fallback_reason": "insufficient_context",
    }


def test_insufficient_context_not_triggered_as_substring() -> None:
    # sentinel embedded mid-string should NOT trigger — prefix check only
    answer = f"This is fine. {INSUFFICIENT_CONTEXT} appeared in context."
    node, _ = _node(answer)
    state = {
        "retrieved_docs": [_doc("Some context.")],
        "rewritten_query": "A question",
    }
    result = node(state)  # type: ignore[arg-type]
    assert result == {"english_response": answer}


# LLM exception → llm_error fallback
@pytest.mark.parametrize(
    "exc",
    [RuntimeError("provider down"), OSError("timeout"), Exception("boom")],
)
def test_llm_exception_returns_llm_error_fallback(exc: Exception) -> None:
    node, _ = _node(side_effect=exc)
    state = {
        "retrieved_docs": [_doc("PM Kisan scheme context.")],
        "rewritten_query": "What is PM Kisan?",
    }
    result = node(state)  # type: ignore[arg-type]
    assert result == {
        "english_response": "",
        "fallback_triggered": True,
        "fallback_reason": "llm_error",
    }


# _strip_thinking helper
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<think>reasoning here</think>The answer.", "The answer."),
        ("<THINK>reasoning</THINK>The answer.", "The answer."),
        ("<|think|>reasoning<br/>details</|think|>The answer.", "The answer."),
        ("No thinking tags here.", "No thinking tags here."),
        ("<think>multi\nline\nreasoning</think>\nClean answer.", "Clean answer."),
    ],
    ids=[
        "lowercase-think",
        "uppercase-think",
        "pipe-delimited-think",
        "no-tags",
        "multiline-think",
    ],
)
def test_strip_thinking(raw: str, expected: str) -> None:
    assert _strip_thinking(raw) == expected


# _build_context helper
@pytest.mark.parametrize(
    ("docs", "expected"),
    [
        ([], ""),
        ([_doc("PM Kisan gives ₹6000/year.")], "[1] PM Kisan gives ₹6000/year."),
        (
            [_doc("First chunk."), _doc("Second chunk.")],
            "[1] First chunk.\n\n---\n\n[2] Second chunk.",
        ),
    ],
    ids=["empty", "single-doc", "multiple-docs-numbered-and-separated"],
)
def test_build_context(docs: list, expected: str) -> None:
    assert _build_context(docs) == expected

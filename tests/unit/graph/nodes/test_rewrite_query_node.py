from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.nodes.rewrite_query import (
    _build_history_context,
    _format_recent,
    make_rewrite_query,
)


def _node(llm: MagicMock | None = None):
    deps = MagicMock()
    if llm is not None:
        deps.llm = llm
    return make_rewrite_query(deps)


def _llm_returning(text: str) -> MagicMock:
    llm = MagicMock()
    resp = MagicMock()
    resp.content = text
    llm.complete.return_value = resp
    return llm


# early-exit paths - LLM never called
@pytest.mark.parametrize(
    ("state", "expected_rewritten"),
    [
        (
            {
                "english_query": "What is PM Kisan?",
                "messages": [],
                "conversation_summary": "",
            },
            "What is PM Kisan?",
        ),
        (
            {"english_query": "How do I apply?"},
            "How do I apply?",
        ),
        (
            {
                "english_query": "Tell me more.",
                "messages": None,
                "conversation_summary": None,
            },
            "Tell me more.",
        ),
    ],
    ids=[
        "empty-messages-no-summary",
        "missing-messages-key",
        "none-messages-none-summary",
    ],
)
def test_no_history_skips_llm(state: dict, expected_rewritten: str) -> None:
    llm = MagicMock()
    result = _node(llm)(state)
    llm.complete.assert_not_called()
    assert result == {"rewritten_query": expected_rewritten}


@pytest.mark.parametrize(
    "query",
    ["", "   ", "\t\n"],
    ids=["empty", "whitespace-only", "tab-newline"],
)
def test_blank_query_skips_llm(query: str) -> None:
    llm = MagicMock()
    state = {
        "english_query": query,
        "messages": [HumanMessage(content="Previous question")],
        "conversation_summary": "",
    }
    result = _node(llm)(state)
    llm.complete.assert_not_called()
    assert result == {"rewritten_query": query}


# LLM called - rewrite returned
@pytest.mark.parametrize(
    ("state", "llm_response", "expected_rewritten"),
    [
        (
            {
                "english_query": "How do I apply?",
                "messages": [
                    HumanMessage(content="What is PM Kisan?"),
                    AIMessage(content="PM Kisan gives ₹6000/year."),
                ],
                "conversation_summary": "",
            },
            "How do I apply for PM Kisan?",
            "How do I apply for PM Kisan?",
        ),
        (
            {
                "english_query": "What about eligibility?",
                "messages": [],
                "conversation_summary": "User asked about PM Kisan scheme and its benefits.",
            },
            "What is the eligibility for PM Kisan scheme?",
            "What is the eligibility for PM Kisan scheme?",
        ),
        (
            {
                "english_query": "Is it still active?",
                "messages": [
                    HumanMessage(content="Tell me about Ayushman Bharat."),
                    AIMessage(content="Ayushman Bharat provides health cover."),
                ],
                "conversation_summary": "Earlier the user discussed PM Kisan benefits.",
            },
            "Is Ayushman Bharat still active?",
            "Is Ayushman Bharat still active?",
        ),
    ],
    ids=["messages-only", "summary-only", "messages-and-summary"],
)
def test_rewrite_calls_llm_and_returns_result(
    state: dict, llm_response: str, expected_rewritten: str
) -> None:
    llm = _llm_returning(llm_response)
    result = _node(llm)(state)
    llm.complete.assert_called_once()
    assert result == {"rewritten_query": expected_rewritten}


# LLM edge cases
@pytest.mark.parametrize(
    ("llm_response", "english_query"),
    [
        ("", "How do I register?"),
        ("   ", "What documents are needed?"),
    ],
    ids=["empty-response", "whitespace-response"],
)
def test_llm_blank_response_falls_back_to_original(
    llm_response: str, english_query: str
) -> None:
    llm = _llm_returning(llm_response)
    state = {
        "english_query": english_query,
        "messages": [HumanMessage(content="What is PM Kisan?")],
        "conversation_summary": "",
    }
    result = _node(llm)(state)
    assert result == {"rewritten_query": english_query}


@pytest.mark.parametrize(
    "exc",
    [RuntimeError("provider down"), OSError("timeout"), Exception("boom")],
)
def test_llm_exception_falls_back_to_original(exc: Exception) -> None:
    llm = MagicMock()
    llm.complete.side_effect = exc
    state = {
        "english_query": "What is the benefit amount?",
        "messages": [HumanMessage(content="Tell me about PM Kisan.")],
        "conversation_summary": "",
    }
    result = _node(llm)(state)
    assert result == {"rewritten_query": "What is the benefit amount?"}


# _format_recent
@pytest.mark.parametrize(
    ("messages", "expected_lines"),
    [
        (
            [
                HumanMessage(content="What is PM Kisan?"),
                AIMessage(content="PM Kisan pays ₹6000."),
            ],
            ["User: What is PM Kisan?", "Assistant: PM Kisan pays ₹6000."],
        ),
        (
            [HumanMessage(content="Hello")],
            ["User: Hello"],
        ),
        (
            [],
            [],
        ),
    ],
    ids=["human-then-ai", "human-only", "empty"],
)
def test_format_recent(messages: list, expected_lines: list[str]) -> None:
    result = _format_recent(messages)
    assert result == "\n".join(expected_lines)


# _build_history_context
@pytest.mark.parametrize(
    ("messages", "summary", "expected"),
    [
        (
            [],
            "User asked about PM Kisan.",
            "[Summary of earlier conversation]\nUser asked about PM Kisan.",
        ),
        (
            [
                HumanMessage(content="What is PM Kisan?"),
                AIMessage(content="It gives ₹6000."),
            ],
            "",
            "[Recent turns]\nUser: What is PM Kisan?\nAssistant: It gives ₹6000.",
        ),
        (
            [HumanMessage(content="How do I apply?")],
            "User discussed PM Kisan.",
            "[Summary of earlier conversation]\nUser discussed PM Kisan.\n\n[Recent turns]\nUser: How do I apply?",
        ),
        ([], "", ""),
    ],
    ids=["summary-only", "messages-only", "both-summary-and-messages", "empty"],
)
def test_build_history_context(messages: list, summary: str, expected: str) -> None:
    assert _build_history_context(messages, summary) == expected

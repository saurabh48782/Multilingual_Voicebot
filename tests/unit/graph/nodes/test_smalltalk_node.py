from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.nodes.smalltalk import _CANNED, make_smalltalk
from src.graph.prompts import SMALLTALK_PROMPT, SMALLTALK_SYSTEM


def _node(
    llm_content: str | None = None, side_effect: Exception | None = None
) -> tuple[Any, MagicMock]:
    deps = MagicMock()
    resp = MagicMock()
    if side_effect:
        deps.llm.complete.side_effect = side_effect
    else:
        resp.content = llm_content or ""
        deps.llm.complete.return_value = resp
    return make_smalltalk(deps), deps


# successful generation
def test_successful_generation_returns_english_response() -> None:
    answer = "Hi there! How can I help you today?"
    node, deps = _node(answer)
    result = node({"english_query": "Hello"})
    deps.llm.complete.assert_called_once()
    assert result == {"english_response": answer, "intent": "general"}


def test_response_is_stripped() -> None:
    node, _ = _node("  Hello there.  \n")
    result = node({"english_query": "Hi"})
    assert result == {"english_response": "Hello there.", "intent": "general"}


# query preference: rewritten_query > english_query
@pytest.mark.parametrize(
    ("state", "expected_query"),
    [
        (
            {"rewritten_query": "How are you doing?", "english_query": "How is it?"},
            "How are you doing?",
        ),
        (
            {"english_query": "Who are you?"},
            "Who are you?",
        ),
    ],
    ids=["rewritten-query-preferred", "falls-back-to-english-query"],
)
def test_query_preference(state: dict[str, Any], expected_query: str) -> None:
    node, deps = _node("A reply.")
    node(state)
    call = deps.llm.complete.call_args
    assert expected_query in call.kwargs["messages"][-1]["content"]


# system + user prompt wiring
def test_prompt_wiring() -> None:
    node, deps = _node("A reply.")
    node(
        {
            "english_query": "What's your name?",
            "messages": [
                HumanMessage(content="Hi, I'm Ram."),
                AIMessage(content="Hello Ram!"),
            ],
            "conversation_summary": "",
        }
    )
    messages = deps.llm.complete.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": SMALLTALK_SYSTEM}
    user_content = messages[-1]["content"]
    assert (
        SMALLTALK_PROMPT.format(
            history="[Recent turns]\nUser: Hi, I'm Ram.\nAssistant: Hello Ram!",
            query="What's your name?",
        )
        == user_content
    )


# history threading - conversation history reaches the prompt
@pytest.mark.parametrize(
    ("state", "expected_fragment"),
    [
        (
            {
                "english_query": "What did I just say?",
                "messages": [HumanMessage(content="My name is Sita.")],
                "conversation_summary": "",
            },
            "User: My name is Sita.",
        ),
        (
            {
                "english_query": "Continue.",
                "messages": [],
                "conversation_summary": "The user introduced themselves as Gita.",
            },
            "The user introduced themselves as Gita.",
        ),
    ],
    ids=["messages-only", "summary-only"],
)
def test_history_reaches_prompt(state: dict[str, Any], expected_fragment: str) -> None:
    node, deps = _node("Sure.")
    node(state)
    user_content = deps.llm.complete.call_args.kwargs["messages"][-1]["content"]
    assert expected_fragment in user_content


@pytest.mark.parametrize(
    "state",
    [
        {"english_query": "Hi", "messages": [], "conversation_summary": ""},
        {"english_query": "Hi", "messages": None, "conversation_summary": None},
        {"english_query": "Hi"},
    ],
    ids=["empty-history", "none-history", "missing-history-keys"],
)
def test_no_history_still_generates(state: dict[str, Any]) -> None:
    node, deps = _node("Hello!")
    result = node(state)
    deps.llm.complete.assert_called_once()
    assert result == {"english_response": "Hello!", "intent": "general"}


# blank LLM response falls back to canned greeting
@pytest.mark.parametrize(
    "llm_response",
    ["", "   ", "\t\n"],
    ids=["empty", "whitespace-only", "tab-newline"],
)
def test_blank_response_falls_back_to_canned(llm_response: str) -> None:
    node, _ = _node(llm_response)
    result = node({"english_query": "Hello"})
    assert result == {"english_response": _CANNED, "intent": "general"}


# LLM exception falls back to canned greeting
@pytest.mark.parametrize(
    "exc",
    [RuntimeError("provider down"), OSError("timeout"), Exception("boom")],
)
def test_llm_exception_falls_back_to_canned(exc: Exception) -> None:
    node, _ = _node(side_effect=exc)
    result = node({"english_query": "Hello"})
    assert result == {"english_response": _CANNED, "intent": "general"}

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from src.graph.nodes.summarize import (
    KEEP_RECENT,
    SUMMARIZE_THRESHOLD,
    _format_messages,
    make_summarize,
    should_summarize,
)


def _human(text: str) -> HumanMessage:
    return HumanMessage(content=text)


def _ai(text: str) -> AIMessage:
    return AIMessage(content=text)


def _messages(n: int) -> list[HumanMessage | AIMessage]:
    return [_human(f"q{i}") if i % 2 == 0 else _ai(f"a{i}") for i in range(n)]


def _node(llm_content: str | None = None, side_effect: Exception | None = None) -> Any:
    deps = MagicMock()
    resp = MagicMock()
    if side_effect:
        deps.llm.complete.side_effect = side_effect
    else:
        resp.content = llm_content or ""
        deps.llm.complete.return_value = resp
    return make_summarize(deps)


# should_summarize router
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({}, "end"),
        ({"messages": []}, "end"),
        ({"messages": _messages(SUMMARIZE_THRESHOLD - 1)}, "end"),
        ({"messages": _messages(SUMMARIZE_THRESHOLD)}, "summarize"),
        ({"messages": _messages(SUMMARIZE_THRESHOLD + 4)}, "summarize"),
    ],
    ids=[
        "no-messages-key",
        "zero-messages",
        "below-threshold",
        "at-threshold",
        "above-threshold",
    ],
)
def test_should_summarize(state: dict[str, Any], expected: str) -> None:
    assert should_summarize(state) == expected  # type: ignore[arg-type]


# make_summarize: nothing to compress
@pytest.mark.parametrize(
    "n_messages",
    [0, 1, KEEP_RECENT],
    ids=["zero", "one", "exactly-keep-recent"],
)
def test_nothing_to_compress_returns_empty_dict(n_messages: int) -> None:
    node = _node("New summary.")
    result = node({"messages": _messages(n_messages), "conversation_summary": ""})
    assert result == {}


# make_summarize: compression
def test_compresses_old_messages_and_keeps_recent() -> None:
    # 10 messages total → 6 compressed (oldest), 4 kept
    messages = _messages(10)
    to_compress = messages[:-KEEP_RECENT]
    node = _node("Summary of older turns.")
    result = node({"messages": messages, "conversation_summary": ""})

    assert result["conversation_summary"] == "Summary of older turns."
    removals = result["messages"]
    assert len(removals) == len(to_compress)
    assert all(isinstance(r, RemoveMessage) for r in removals)
    assert {r.id for r in removals} == {m.id for m in to_compress}


def test_previous_summary_included_in_llm_prompt() -> None:
    deps = MagicMock()
    resp = MagicMock()
    resp.content = "Updated summary."
    deps.llm.complete.return_value = resp
    node = make_summarize(deps)
    node({"messages": _messages(8), "conversation_summary": "Prior summary."})

    call_user_content = deps.llm.complete.call_args.kwargs["messages"][-1]["content"]
    assert "Prior summary." in call_user_content


# LLM failure (exception or empty response) falls back to the previous summary
@pytest.mark.parametrize(
    ("llm_content", "side_effect", "previous_summary"),
    [
        ("", None, "Preserved."),
        (None, RuntimeError("provider down"), "Old summary."),
        (None, OSError("timeout"), "Old summary."),
        (None, Exception("boom"), "Old summary."),
    ],
    ids=["empty-response", "runtime-error", "os-error", "generic-exception"],
)
def test_llm_failure_keeps_previous_summary(
    llm_content: str | None, side_effect: Exception | None, previous_summary: str
) -> None:
    node = _node(llm_content, side_effect)
    result = node({"messages": _messages(8), "conversation_summary": previous_summary})
    if side_effect is not None:
        # An LLM exception must emit no state update at all (no RemoveMessage),
        # so the checkpointed summary + messages are left intact.
        assert result == {}
    else:
        assert result["conversation_summary"] == previous_summary


# _format_messages helper
@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (
            [_human("What is PM Kisan?"), _ai("PM Kisan pays ₹6000.")],
            "User: What is PM Kisan?\nAssistant: PM Kisan pays ₹6000.",
        ),
        ([_human("Hello")], "User: Hello"),
        ([], ""),
    ],
    ids=["human-then-ai", "human-only", "empty"],
)
def test_format_messages(messages: list[HumanMessage | AIMessage], expected: str) -> None:
    assert _format_messages(messages) == expected

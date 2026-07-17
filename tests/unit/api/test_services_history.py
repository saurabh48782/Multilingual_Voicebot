"""Tests for `history_from_state` — session-history replay from the checkpoint.

The key behavior: assistant turns store English `content` (for the coreference
rewrite) but the vernacular reply the user saw lives in
`additional_kwargs["vernacular"]`. Replay must prefer the vernacular copy, and
fall back to `content` for legacy turns saved before that kwarg existed.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.api.services import history_from_state


class _FakeGraph:
    """Minimal stand-in exposing the single method `history_from_state` uses."""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = messages

    async def aget_state(self, _cfg: dict[str, Any]) -> Any:
        return type("Snapshot", (), {"values": {"messages": self._messages}})


@pytest.mark.asyncio
async def test_assistant_replay_prefers_vernacular() -> None:
    graph = _FakeGraph(
        [
            HumanMessage(content="मेरा होमटाउन कहाँ है?"),
            AIMessage(
                content="Saurabh's hometown is Agra.",
                additional_kwargs={"vernacular": "सौरभ का गृह नगर आगरा है।", "language": "hi"},
            ),
        ]
    )
    history = await history_from_state(graph, "sess-1")
    assert [(m.role, m.content) for m in history.messages] == [
        ("user", "मेरा होमटाउन कहाँ है?"),
        ("assistant", "सौरभ का गृह नगर आगरा है।"),
    ]


@pytest.mark.asyncio
async def test_assistant_replay_falls_back_to_english_content() -> None:
    # Legacy turn: no vernacular kwarg -> replay the English content as-is.
    graph = _FakeGraph(
        [
            HumanMessage(content="सवाल"),
            AIMessage(content="PM Kisan gives farmers ₹6000 a year."),
        ]
    )
    history = await history_from_state(graph, "sess-2")
    assert history.messages[1].content == "PM Kisan gives farmers ₹6000 a year."


@pytest.mark.asyncio
async def test_empty_state_returns_no_messages() -> None:
    graph = _FakeGraph([])
    history = await history_from_state(graph, "sess-3")
    assert history.session_id == "sess-3"
    assert history.messages == []

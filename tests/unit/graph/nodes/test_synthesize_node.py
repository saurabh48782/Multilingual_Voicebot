from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.nodes.synthesize import make_synthesize
from src.graph.state import VoicebotState


def _node(
    tts: MagicMock | None = None,
) -> tuple[Callable[[VoicebotState], dict[str, Any]], MagicMock]:
    deps = MagicMock()
    if tts is not None:
        deps.tts = tts
    deps.audio_cache.put.return_value = "cached-audio-id"
    return make_synthesize(deps), deps


# TTS called / not called
def test_tts_called_when_text_present() -> None:
    tts = MagicMock()
    tts.synthesize.return_value = b"AUDIO"
    state = {
        "vernacular_response": "पीएम किसान ₹6000 देता है।",
        "source_language": "hi",
        "input_mode": "voice",
    }
    node, deps = _node(tts)
    result = node(state)  # type: ignore[arg-type]
    tts.synthesize.assert_called_once_with("पीएम किसान ₹6000 देता है।", "hi")
    deps.audio_cache.put.assert_called_once_with(b"AUDIO", content_type="audio/wav")
    assert result["audio_id"] == "cached-audio-id"


def test_tts_skipped_for_text_input() -> None:
    # Text-in turn: reply text is present but we must not synthesize audio.
    tts = MagicMock()
    tts.synthesize.return_value = b"AUDIO"
    state = {
        "vernacular_response": "PM Kisan gives ₹6000.",
        "source_language": "en",
        "input_mode": "text",
        "transcript": "What is PM Kisan?",
        "english_response": "PM Kisan gives ₹6000.",
    }
    node, deps = _node(tts)
    result = node(state)  # type: ignore[arg-type]
    tts.synthesize.assert_not_called()
    deps.audio_cache.put.assert_not_called()
    assert result["audio_id"] is None
    # The turn is still recorded to memory even without audio.
    assert len(result["messages"]) == 2


@pytest.mark.parametrize(
    "state",
    [
        {"vernacular_response": "", "source_language": "en"},
        {"source_language": "en"},
    ],
    ids=["empty-text", "missing-key"],
)
def test_tts_not_called_when_text_absent(state: dict[str, Any]) -> None:
    tts = MagicMock()
    node, deps = _node(tts)
    result = node(state)  # type: ignore[arg-type]
    tts.synthesize.assert_not_called()
    deps.audio_cache.put.assert_not_called()
    assert result["audio_id"] is None


# TTS exception - graceful degradation
@pytest.mark.parametrize(
    "exc",
    [RuntimeError("sidecar down"), OSError("timeout"), Exception("boom")],
)
def test_tts_exception_returns_none_audio_but_still_appends_messages(
    exc: Exception,
) -> None:
    tts = MagicMock()
    tts.synthesize.side_effect = exc
    state = {
        "vernacular_response": "PM Kisan gives ₹6000.",
        "source_language": "en",
        "input_mode": "voice",
        "transcript": "What is PM Kisan?",
        "english_response": "PM Kisan gives ₹6000.",
    }
    node, deps = _node(tts)
    result = node(state)  # type: ignore[arg-type]
    deps.audio_cache.put.assert_not_called()
    assert result["audio_id"] is None
    assert len(result["messages"]) == 2


# always returns audio_content_type
def test_audio_content_type_always_set() -> None:
    node, _ = _node()
    result = node({"vernacular_response": "", "source_language": "en"})
    assert result["audio_content_type"] == "audio/wav"


# message construction
@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("Scrubbed question.", "Scrubbed question."),
        (None, ""),
    ],
    ids=["transcript-present", "transcript-absent"],
)
def test_messages_human_turn(transcript: str | None, expected: str) -> None:
    state = {
        "vernacular_response": "Answer.",
        "source_language": "en",
        "english_response": "Answer.",
    }
    if transcript is not None:
        state["transcript"] = transcript
    node, _ = _node()
    result = node(state)  # type: ignore[arg-type]
    human_msg = result["messages"][0]
    assert isinstance(human_msg, HumanMessage)
    assert human_msg.content == expected


@pytest.mark.parametrize(
    ("english_response", "vernacular_response", "expected"),
    [
        ("PM Kisan gives ₹6000.", "पीएम किसान।", "PM Kisan gives ₹6000."),
        (None, "পিএম কিষান।", "পিএম কিষান।"),
    ],
    ids=["english-response-present", "english-response-absent"],
)
def test_messages_ai_turn(
    english_response: str | None, vernacular_response: str, expected: str
) -> None:
    state = {
        "vernacular_response": vernacular_response,
        "source_language": "hi",
        "transcript": "सवाल",
    }
    if english_response is not None:
        state["english_response"] = english_response
    node, _ = _node()
    result = node(state)  # type: ignore[arg-type]
    ai_msg = result["messages"][1]
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == expected


@pytest.mark.parametrize(
    "lang",
    ["hi", "bn", "en"],
    ids=["hindi", "bengali", "english"],
)
def test_tts_receives_correct_language(lang: str) -> None:
    tts = MagicMock()
    tts.synthesize.return_value = b"AUDIO"
    state = {
        "vernacular_response": "Some text.",
        "source_language": lang,
        "input_mode": "voice",
    }
    node, _ = _node(tts)
    node(state)  # type: ignore[arg-type]
    tts.synthesize.assert_called_once_with("Some text.", lang)

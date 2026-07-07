"""Tests call make_transcribe(deps) directly - no graph compile, no other nodes.
deps is a MagicMock so only deps.stt is ever touched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.graph.nodes.transcribe import make_transcribe
from src.graph.state import VoicebotState
from src.stt.base import TranscriptionResult


def _node(stt: MagicMock | None = None) -> Callable[[VoicebotState], dict[str, Any]]:
    deps = MagicMock()
    if stt is not None:
        deps.stt = stt
    return make_transcribe(deps)


# Text-only paths — STT never called
RESET_FLAGS = {"fallback_triggered": False, "fallback_reason": "", "grounded": False}


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {"requested_language": "hi", "text_input": "नमस्ते"},
            {"transcript": "नमस्ते", "source_language": "hi", **RESET_FLAGS},
        ),
        (
            {"requested_language": "en"},
            {"transcript": "", "source_language": "en", **RESET_FLAGS},
        ),
        (
            {
                "requested_language": "bn",
                "text_input": "",
            },  # empty string treated as absent
            {"transcript": "", "source_language": "bn", **RESET_FLAGS},
        ),
    ],
    ids=["text-passthrough", "no-input", "empty-string-absent"],
)
def test_text_only_skips_stt(state: dict[str, Any], expected: dict[str, Any]) -> None:
    stt = MagicMock()
    assert _node(stt)(state) == expected  # type: ignore[arg-type]
    stt.transcribe.assert_not_called()


# Audio paths — STT called, result propagated
@pytest.mark.parametrize(
    ("state", "stt_result", "expected_transcript", "expected_language"),
    [
        (
            {"requested_language": "en", "audio_input": b"BYTES"},
            TranscriptionResult(text="hello", language="en"),
            "hello",
            "en",
        ),
        (
            {"requested_language": "hi", "audio_input": b"AUDIOBYTES"},
            TranscriptionResult(text="पीएम किसान योजना क्या है?", language="hi"),
            "पीएम किसान योजना क्या है?",
            "hi",
        ),
        (
            # STT may normalise the language tag; its output wins over requested_language
            {"requested_language": "hi", "audio_input": b"BYTES"},
            TranscriptionResult(text="hello", language="en"),
            "hello",
            "en",
        ),
    ],
    ids=["english-audio", "hindi-audio", "stt-language-overrides-requested"],
)
def test_audio_calls_stt_and_returns_result(
    state: dict[str, Any],
    stt_result: TranscriptionResult,
    expected_transcript: str,
    expected_language: str,
) -> None:
    stt = MagicMock()
    stt.transcribe.return_value = stt_result
    result = _node(stt)(state)  # type: ignore[arg-type]

    stt.transcribe.assert_called_once_with(
        state["audio_input"], language=state["requested_language"]
    )
    assert result["transcript"] == expected_transcript
    assert result["source_language"] == expected_language
    assert result["fallback_triggered"] is False


# STT failure — graceful degradation
@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("provider down"),
        OSError("timeout"),
        ValueError("bad audio"),
        Exception("boom"),
    ],
)
def test_stt_exception_triggers_fallback(exc: Exception) -> None:
    stt = MagicMock()
    stt.transcribe.side_effect = exc
    result = _node(stt)({"requested_language": "hi", "audio_input": b"X"})

    assert result == {
        "audio_input": None,
        "transcript": "",
        "source_language": "hi",
        "fallback_triggered": True,
        "fallback_reason": "stt_error",
        "grounded": False,
    }


def test_resets_sticky_fallback_flags_from_prior_turn() -> None:
    """C1 regression: a fallback in turn N must not poison turn N+1.

    The checkpointer carries fallback_triggered/fallback_reason/grounded
    forward across turns, so transcribe - as the first node of every turn -
    must reset them regardless of this turn's own outcome.
    """
    stt = MagicMock()
    stt.transcribe.return_value = TranscriptionResult(text="hello", language="en")
    stale_state = {
        "requested_language": "en",
        "audio_input": b"BYTES",
        "fallback_triggered": True,
        "fallback_reason": "verifier_error",
        "grounded": False,
    }

    result = _node(stt)(stale_state)  # type: ignore[arg-type]

    assert result["fallback_triggered"] is False
    assert result["fallback_reason"] == ""
    assert result["grounded"] is False

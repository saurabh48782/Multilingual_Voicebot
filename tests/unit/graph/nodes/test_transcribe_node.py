"""Tests call make_transcribe(deps) directly - no graph compile, no other nodes.
deps is a MagicMock so only deps.stt is ever touched.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.graph.nodes.transcribe import make_transcribe
from src.stt.base import TranscriptionResult


def _node(stt: MagicMock | None = None):
    deps = MagicMock()
    if stt is not None:
        deps.stt = stt
    return make_transcribe(deps)


# Text-only paths — STT never called
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {"requested_language": "hi", "text_input": "नमस्ते"},
            {"transcript": "नमस्ते", "source_language": "hi"},
        ),
        (
            {"requested_language": "en"},
            {"transcript": "", "source_language": "en"},
        ),
        (
            {
                "requested_language": "bn",
                "text_input": "",
            },  # empty string treated as absent
            {"transcript": "", "source_language": "bn"},
        ),
    ],
    ids=["text-passthrough", "no-input", "empty-string-absent"],
)
def test_text_only_skips_stt(state: dict, expected: dict) -> None:
    stt = MagicMock()
    assert _node(stt)(state) == expected
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
    state: dict,
    stt_result: TranscriptionResult,
    expected_transcript: str,
    expected_language: str,
) -> None:
    stt = MagicMock()
    stt.transcribe.return_value = stt_result
    result = _node(stt)(state)

    stt.transcribe.assert_called_once_with(
        state["audio_input"], language=state["requested_language"]
    )
    assert result["transcript"] == expected_transcript
    assert result["source_language"] == expected_language
    assert "fallback_triggered" not in result


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
        "transcript": "",
        "source_language": "hi",
        "fallback_triggered": True,
        "fallback_reason": "stt_error",
    }

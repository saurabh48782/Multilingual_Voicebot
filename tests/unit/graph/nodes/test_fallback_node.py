from __future__ import annotations

import pytest

from src.graph.nodes.fallback import fallback
from src.graph.prompts import FALLBACK_MESSAGES


@pytest.mark.parametrize(
    ("state", "expected_vernacular"),
    [
        ({"source_language": "hi"}, FALLBACK_MESSAGES["hi"]),
        ({"source_language": "bn"}, FALLBACK_MESSAGES["bn"]),
        ({"source_language": "en"}, FALLBACK_MESSAGES["en"]),
        ({"source_language": "unknown_lang"}, FALLBACK_MESSAGES["en"]),
        ({}, FALLBACK_MESSAGES["en"]),
    ],
    ids=[
        "hindi",
        "bengali",
        "english",
        "unknown-lang-falls-back-to-english",
        "missing-source-language-defaults-english",
    ],
)
def test_fallback_response(state: dict[str, str], expected_vernacular: str) -> None:
    result = fallback(state)  # type: ignore[arg-type]
    assert result["vernacular_response"] == expected_vernacular
    assert result["english_response"] == FALLBACK_MESSAGES["en"]
    assert result["fallback_triggered"] is True


@pytest.mark.parametrize(
    ("state_reason", "expected_reason"),
    [
        ({"source_language": "en", "fallback_reason": "stt_error"}, "stt_error"),
        (
            {"source_language": "en", "fallback_reason": "retrieval_error"},
            "retrieval_error",
        ),
        ({"source_language": "en"}, "retrieval_gate"),
        ({"source_language": "en", "fallback_reason": None}, "retrieval_gate"),
    ],
    ids=[
        "preserves-stt-error",
        "preserves-retrieval-error",
        "defaults-to-gate",
        "none-defaults-to-gate",
    ],
)
def test_fallback_reason(state_reason: dict[str, str | None], expected_reason: str) -> None:
    result = fallback(state_reason)  # type: ignore[arg-type]
    assert result["fallback_reason"] == expected_reason

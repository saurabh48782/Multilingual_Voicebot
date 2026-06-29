from __future__ import annotations

import pytest

from src.graph.nodes.pii_scrub import pii_scrub


@pytest.mark.parametrize(
    ("transcript", "contains_mask"),
    [
        ("My Aadhaar is 1234 5678 9012 and I need help.", "[AADHAAR]"),
        ("PAN card: ABCDE1234F for verification.", "[PAN]"),
        ("Call me on 9876543210 for details.", "[PHONE]"),
    ],
    ids=["aadhaar", "pan", "phone"],
)
def test_pii_patterns_are_masked(transcript: str, contains_mask: str) -> None:
    result = pii_scrub({"transcript": transcript})  # type: ignore[arg-type]
    assert contains_mask in result["transcript"]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {"transcript": "What is the PM Kisan scheme?"},
            "What is the PM Kisan scheme?",
        ),
        ({}, ""),
    ],
    ids=["clean_passthrough", "missing_empty"],
)
def test_transcript_passthrough_and_missing(state: dict, expected: str) -> None:
    result = pii_scrub(state)  # type: ignore[arg-type]
    assert result["transcript"] == expected

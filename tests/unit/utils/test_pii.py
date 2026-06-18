"""Unit tests for PII redaction patterns."""

import pytest

from src.utils.pii import scrub


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", ""),
        ("My Aadhaar is 1234 5678 9012", "My Aadhaar is [AADHAAR]"),
        ("Aadhaar: 123456789012", "Aadhaar: [AADHAAR]"),
        ("PAN card: ABCDE1234F is valid", "PAN card: [PAN] is valid"),
        ("call me at 9876543210", "call me at [PHONE]"),
        ("contact: +919876543210", "contact: [PHONE]"),
        # 9-digit account - unambiguous (not 12-digit Aadhaar, not 10-digit phone)
        ("account no 123456789", "account no [ACCOUNT]"),
        ("email me at test.kumar+pm@example.co.in", "email me at [EMAIL]"),
        # Phone written in Devanagari numerals (9876543210)
        ("मेरा फ़ोन ९८७६५४३२१० है", "मेरा फ़ोन [PHONE] है"),
        ("no pii here just text", "no pii here just text"),
    ],
)
def test_scrub(text: str, expected: str) -> None:
    assert scrub(text) == expected


def test_scrub_multiple_in_one_string() -> None:
    raw = "Aadhaar 1234 5678 9012 and PAN ABCDE1234F and phone 9123456789"
    result = scrub(raw)
    assert "[AADHAAR]" in result
    assert "[PAN]" in result
    assert "[PHONE]" in result
    assert "1234 5678 9012" not in result
    assert "ABCDE1234F" not in result  # pragma: allowlist secret
    assert "9123456789" not in result

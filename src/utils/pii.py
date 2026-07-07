"""PII redaction patterns. Used by the pii_scrub graph node before transcripts reach logs or LLMs.

Notes on coverage:
- `\\d` in Python's str regexes matches any Unicode decimal digit, so Aadhaar /
  account / PAN-digit runs written in Devanagari (०-९) or Bengali (০-৯)
  numerals are caught. Literal character classes like the phone-prefix
  [6-9] are ASCII-only, so the vernacular six-to-nine ranges are added
  explicitly.
- ACCOUNT (9-18 digit runs) is deliberately broad: over-redacting an
  occasional scheme ID is preferred to leaking a bank account number.
"""

from __future__ import annotations

import re

# Digits 6-9 in each supported script, for the Indian mobile prefix.
_SIX_TO_NINE = "6-9६-९৬-৯௬-௯"

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    # PHONE must precede AADHAAR and ACCOUNT - a 10-digit Indian phone matches
    # both \d{9,18} and (with +91 prefix) \d{4}\s?\d{4}\s?\d{4}. Digits may be
    # grouped with spaces/hyphens (e.g. "98765 43210", "9876-543-210").
    (
        re.compile(rf"(?<!\d)(?:\+91[\s-]?|91[\s-]?)?" rf"[{_SIX_TO_NINE}](?:[\s-]?\d){{9}}(?!\d)"),
        "[PHONE]",
    ),
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[AADHAAR]"),
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE), "[PAN]"),
    (re.compile(r"\b\d{9,18}\b"), "[ACCOUNT]"),
]


def scrub(text: str) -> str:
    """Return text with all PII replaced by placeholder tokens."""
    for pattern, placeholder in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text

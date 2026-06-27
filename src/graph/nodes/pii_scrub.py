"""Mask Aadhaar / PAN / phone / account numbers before they hit logs or LLMs."""

from __future__ import annotations

from typing import Any

from src.graph.state import VoicebotState
from src.utils.pii import scrub


def pii_scrub(state: VoicebotState) -> dict[str, Any]:
    transcript = state.get("transcript", "")
    scrubbed = scrub(transcript)
    return {"transcript_scrubbed": scrubbed, "transcript": scrubbed}

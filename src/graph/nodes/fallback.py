"""Canned vernacular "no info" message. Bypasses LLM + translator."""

from __future__ import annotations

from typing import Any

from src.graph.prompts import FALLBACK_MESSAGES
from src.graph.state import VoicebotState


def fallback(state: VoicebotState) -> dict[str, Any]:
    lang = state.get("source_language", "en")
    message = FALLBACK_MESSAGES.get(lang, FALLBACK_MESSAGES["en"])
    return {
        "vernacular_response": message,
        "english_response": FALLBACK_MESSAGES["en"],
        "fallback_triggered": True,
        "fallback_reason": state.get("fallback_reason") or "retrieval_gate",
    }

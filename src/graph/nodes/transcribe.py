"""Transcribe audio → text + source language.

If the input is text-only (text_input set, audio_input empty) this node passes
the text straight through. `requested_language` is always set and validated
before the graph runs (hi | bn | en). STT failure degrades to an empty
transcript + fallback flag; the empty-query guard in retrieve routes it to
fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.graph.deps import Deps
from src.graph.state import VoicebotState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_transcribe(deps: Deps) -> Callable[[VoicebotState], dict[str, Any]]:
    def transcribe(state: VoicebotState) -> dict[str, Any]:
        audio = state.get("audio_input")
        text = state.get("text_input")

        if not audio:
            if not text:
                return {
                    "transcript": "",
                    "source_language": state["requested_language"],
                }
            return {
                "transcript": text,
                "source_language": state["requested_language"],
            }

        try:
            result = deps.stt.transcribe(audio, language=state["requested_language"])
        except Exception:
            logger.exception("STT provider failed - degrading to fallback")
            return {
                "transcript": "",
                "source_language": state["requested_language"],
                "fallback_triggered": True,
                "fallback_reason": "stt_error",
            }
        return {
            "transcript": result.text,
            "source_language": result.language,
        }

    return transcribe

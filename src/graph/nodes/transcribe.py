"""Transcribe audio → text + source language.

If the input is text-only (text_input set, audio_input empty) this node passes
the text straight through. `requested_language` is always set and validated
before the graph runs (hi | bn | en). STT failure degrades to an empty
transcript + fallback flag; the empty-query guard in retrieve routes it to
fallback.

`audio_input` is cleared from state once consumed here - every later node's
partial update is merged into the running state, so a raw audio blob left in
state would get re-checkpointed by AsyncPostgresSaver on every remaining
super-step of the turn.

As the first node of every turn, this also resets the routing flags
(`fallback_triggered`, `fallback_reason`, `grounded`) that a prior turn may
have left `True`/set in the checkpointed state - otherwise a single fallback
turn would poison every subsequent turn in the session.
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
        # Record how this turn arrived while audio_input is still present (it's
        # cleared below). synthesize reads input_mode to decide whether to speak
        # the reply - text-in turns stay text-only.
        reset_flags = {
            "fallback_triggered": False,
            "fallback_reason": "",
            "grounded": False,
            "input_mode": "voice" if audio else "text",
        }

        if not audio:
            if not text:
                return {
                    "transcript": "",
                    "source_language": state["requested_language"],
                    **reset_flags,
                }
            return {
                "transcript": text,
                "source_language": state["requested_language"],
                **reset_flags,
            }

        try:
            result = deps.stt.transcribe(audio, language=state["requested_language"])
        except Exception:
            logger.exception("STT provider failed - degrading to fallback")
            return {
                "audio_input": None,
                "transcript": "",
                "source_language": state["requested_language"],
                **reset_flags,
                "fallback_triggered": True,
                "fallback_reason": "stt_error",
            }
        return {
            "audio_input": None,
            "transcript": result.text,
            "source_language": result.language,
            **reset_flags,
        }

    return transcribe

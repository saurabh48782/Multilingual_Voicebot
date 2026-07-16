"""Render vernacular response text to speech + append turn to memory.

The synthesized WAV is pushed straight to the `AudioCache` and only its id
is kept in graph state - keeping raw audio bytes out of state means they
never get serialized into the LangGraph checkpoint."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.deps import Deps
from src.graph.state import VoicebotState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_synthesize(deps: Deps) -> Callable[[VoicebotState], dict[str, Any]]:
    def synthesize(state: VoicebotState) -> dict[str, Any]:
        text = state.get("vernacular_response", "")
        lang = state.get("source_language", "en")
        # Speak the reply back only when the turn came in as voice.
        speak = state.get("input_mode") == "voice"
        audio_id: str | None = None
        content_type = "audio/wav"
        if text and speak:
            try:
                audio = deps.tts.synthesize(text, lang)
                audio_id = deps.audio_cache.put(audio, content_type=content_type)
            except Exception:
                logger.exception("TTS failed - returning text-only response")

        new_messages = [
            HumanMessage(content=state.get("transcript", "")),
            AIMessage(
                content=state.get("english_response") or text,
                additional_kwargs={"vernacular": text, "language": lang},
            ),
        ]
        return {
            "audio_id": audio_id,
            "audio_content_type": content_type,
            "messages": new_messages,
        }

    return synthesize

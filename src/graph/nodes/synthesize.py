"""Render vernacular response text to speech bytes + append turn to memory."""

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
        audio: bytes | None = None
        if text:
            try:
                audio = deps.tts.synthesize(text, lang)
            except Exception:
                logger.exception("TTS failed - returning text-only response")

        new_messages = [
            HumanMessage(content=state.get("transcript", "")),
            AIMessage(content=state.get("english_response") or text),
        ]
        return {
            "audio_output": audio,
            "audio_content_type": "audio/wav",
            "messages": new_messages,
        }

    return synthesize

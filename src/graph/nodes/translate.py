"""Translation nodes: vernacular→English (pre-retrieval) and English→vernacular (post-gen).

Translation failures degrade instead of crashing the turn:
- to_english: fall back to the untranslated text - the e5 embedder is
  multilingual, so retrieval still has a chance.
- to_vernacular: fall back to the English answer rather than losing it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.graph.deps import Deps
from src.graph.state import VoicebotState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_to_english(deps: Deps) -> Callable[[VoicebotState], dict[str, Any]]:
    def translate_to_english(state: VoicebotState) -> dict[str, Any]:
        text = state.get("transcript", "")
        lang = state.get("source_language", "en")
        try:
            result = deps.translator.to_english(text, source_language=lang)
        except Exception:
            logger.exception("to_english translation failed - using original text")
            return {"english_query": text}
        return {"english_query": result.text}

    return translate_to_english


def make_to_vernacular(deps: Deps) -> Callable[[VoicebotState], dict[str, Any]]:
    def translate_to_vernacular(state: VoicebotState) -> dict[str, Any]:
        english = state.get("english_response", "")
        lang = state.get("source_language", "en")
        if lang == "en":
            return {"vernacular_response": english}
        try:
            result = deps.translator.to_vernacular(english, target_language=lang)
        except Exception:
            logger.exception("to_vernacular translation failed - replying in English")
            return {"vernacular_response": english}
        return {"vernacular_response": result.text}

    return translate_to_vernacular

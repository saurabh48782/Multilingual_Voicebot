"""Answer `general` chitchat directly, bypassing retrieval + grounding.

Only reached when `classify_intent` labels the turn `general`. Produces an
English response (translated downstream by `translate_to_vernacular`, kept in
English in memory by `synthesize` for coreference), never factual scheme claims.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.graph.deps import Deps
from src.graph.prompts import SMALLTALK_PROMPT, SMALLTALK_SYSTEM
from src.graph.state import VoicebotState
from src.utils.config import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CANNED = "Hello! I can help you with government schemes. What would you like to know?"


def make_smalltalk(deps: Deps) -> Callable[[VoicebotState], dict[str, Any]]:
    def smalltalk(state: VoicebotState) -> dict[str, Any]:
        query = state.get("rewritten_query") or state.get("english_query", "")
        try:
            resp = deps.llm.complete(
                messages=[
                    {"role": "system", "content": SMALLTALK_SYSTEM},
                    {"role": "user", "content": SMALLTALK_PROMPT.format(query=query)},
                ],
                model=cfg["llm"]["model"],
                temperature=0.3,
            )
            answer = resp.content.strip() or _CANNED
        except Exception:
            logger.exception("Small-talk generation failed - using canned greeting")
            answer = _CANNED
        return {"english_response": answer, "intent": "general"}

    return smalltalk

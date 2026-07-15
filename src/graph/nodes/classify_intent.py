"""Classify the turn as `general` chitchat vs `scheme` question.

`general` turns are answered directly by the `smalltalk` node and skip the whole
RAG pipeline (retrieve/rerank/generate/verify). Classification runs after
`rewrite_query`, so coreference-resolved follow-ups ("tell me more") carry their
scheme topic and classify as `scheme`. Empty queries and classifier errors both
default to `scheme` so nothing that needs grounding slips into small talk.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.graph.deps import Deps
from src.graph.prompts import CLASSIFY_PROMPT, CLASSIFY_SYSTEM
from src.graph.state import VoicebotState
from src.utils.config import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_classify_intent(deps: Deps) -> Callable[[VoicebotState], dict[str, Any]]:
    def classify_intent(state: VoicebotState) -> dict[str, Any]:
        # Kill-switch: missing key => enabled (params.yaml may be trimmed).
        if not cfg.get("intent", {}).get("enabled", True):
            return {"intent": "scheme"}

        query = state.get("rewritten_query") or state.get("english_query", "")
        # Empty query: let `retrieve` emit its existing empty_query fallback.
        if not query.strip():
            return {"intent": "scheme"}

        try:
            resp = deps.llm.complete(
                messages=[
                    {"role": "system", "content": CLASSIFY_SYSTEM},
                    {"role": "user", "content": CLASSIFY_PROMPT.format(query=query)},
                ],
                model=cfg["llm"]["model"],
                temperature=0.0,  # deterministic classification
            )
        except Exception:
            logger.exception("Intent classification failed - defaulting to scheme")
            return {"intent": "scheme"}

        intent = "general" if resp.content.strip().upper().startswith("GENERAL") else "scheme"
        return {"intent": intent}

    return classify_intent


def route_after_classify(state: VoicebotState) -> str:
    """Route general chitchat to `smalltalk`, everything else into retrieval."""
    return "smalltalk" if state.get("intent") == "general" else "retrieve"

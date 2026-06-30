"""FAISS retrieval with confidence-gate metadata stored on state.

Empty queries (failed STT, blank input) and retriever errors short-circuit to
the fallback path via retrieval_passed=False instead of searching or crashing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.graph.deps import Deps
from src.graph.state import VoicebotState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_retrieve(deps: Deps) -> Callable[[VoicebotState], dict[str, Any]]:
    def retrieve(state: VoicebotState) -> dict[str, Any]:
        query = state.get("rewritten_query") or state.get("english_query", "")
        if not query.strip():
            return {
                "retrieved_docs": [],
                "retrieval_confidence": 0.0,
                "retrieval_gap": 0.0,
                "retrieval_passed": False,
                "fallback_reason": state.get("fallback_reason") or "empty_query",
            }

        try:
            result = deps.retriever.search(query)
        except Exception:
            logger.exception("Retriever failed - degrading to fallback")
            return {
                "retrieved_docs": [],
                "retrieval_confidence": 0.0,
                "retrieval_gap": 0.0,
                "retrieval_passed": False,
                "fallback_reason": "retrieval_error",
            }
        return {
            "retrieved_docs": result.docs,
            "retrieval_confidence": result.top_score,
            "retrieval_gap": result.gap,
            "retrieval_passed": result.passed,
        }

    return retrieve

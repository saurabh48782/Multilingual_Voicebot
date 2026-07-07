"""Coreference-resolve the English query against prior conversation turns.

If `messages` is empty, the rewritten query == english_query. Otherwise we ask
the LLM to rewrite the query so it is self-contained.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.graph.deps import Deps
from src.graph.prompts import REWRITE_PROMPT, REWRITE_SYSTEM, sanitize_untrusted
from src.graph.state import VoicebotState
from src.utils.config import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _format_recent(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for msg in messages:  # summarize node keeps this bounded to KEEP_RECENT
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {sanitize_untrusted(str(msg.content))}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {sanitize_untrusted(str(msg.content))}")
    return "\n".join(lines)


def _build_history_context(messages: list[BaseMessage], summary: str) -> str:
    parts: list[str] = []
    if summary:
        parts.append(f"[Summary of earlier conversation]\n{sanitize_untrusted(summary)}")
    recent = _format_recent(messages)
    if recent:
        parts.append(f"[Recent turns]\n{recent}")
    return "\n\n".join(parts)


def make_rewrite_query(deps: Deps) -> Callable[[VoicebotState], dict[str, Any]]:
    def rewrite_query(state: VoicebotState) -> dict[str, Any]:
        query = state.get("english_query", "")
        messages = state.get("messages") or []
        summary = state.get("conversation_summary") or ""

        if not messages and not summary:
            return {"rewritten_query": query}
        if not query.strip():
            return {"rewritten_query": query}

        history_context = _build_history_context(list(messages), summary)
        try:
            resp = deps.llm.complete(
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {
                        "role": "user",
                        "content": REWRITE_PROMPT.format(history=history_context, query=query),
                    },
                ],
                model=cfg["llm"]["model"],
                temperature=0.0,  # deterministic coreference resolution
            )
        except Exception:
            logger.exception("Query rewrite failed - using original query")
            return {"rewritten_query": query}
        rewritten = resp.content.strip() or query
        return {"rewritten_query": rewritten}

    return rewrite_query

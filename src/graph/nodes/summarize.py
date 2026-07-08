"""Rolling conversation summarizer.

Triggered after `synthesize` when accumulated messages exceed SUMMARIZE_THRESHOLD.
Compresses the oldest (total - KEEP_RECENT) messages into `conversation_summary`,
then removes those messages from state so the checkpoint stays bounded.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage

from src.graph.deps import Deps
from src.graph.prompts import SUMMARIZE_PROMPT, SUMMARIZE_SYSTEM, sanitize_untrusted
from src.graph.state import VoicebotState
from src.utils.config import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

KEEP_RECENT = 4  # raw messages to retain (last 2 full turns)
SUMMARIZE_THRESHOLD = 8  # total messages before triggering (4+ turns)


def _format_messages(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {sanitize_untrusted(str(msg.content))}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {sanitize_untrusted(str(msg.content))}")
    return "\n".join(lines)


def should_summarize(state: VoicebotState) -> str:
    messages = state.get("messages") or []
    return "summarize" if len(messages) >= SUMMARIZE_THRESHOLD else "end"


def make_summarize(deps: Deps) -> Callable[[VoicebotState], dict[str, Any]]:
    def summarize(state: VoicebotState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        to_compress = messages[:-KEEP_RECENT]
        if not to_compress:
            return {}

        previous_summary = state.get("conversation_summary") or ""
        formatted = _format_messages(to_compress)

        try:
            resp = deps.llm.complete(
                messages=[
                    {"role": "system", "content": SUMMARIZE_SYSTEM},
                    {
                        "role": "user",
                        "content": SUMMARIZE_PROMPT.format(
                            # previous_summary is prior model output derived from
                            # untrusted turns - sanitize so it can't break out of
                            # its <previous_summary> delimiter or forge a sentinel.
                            previous_summary=sanitize_untrusted(previous_summary),
                            history=formatted,
                        ),
                    },
                ],
                model=cfg["llm"]["model"],
                temperature=0.05,  # near-deterministic, faithful compression
            )
            new_summary = resp.content.strip() or previous_summary
        except Exception:
            logger.exception("Summarization failed - keeping messages and summary intact")
            return {}

        removals = [RemoveMessage(id=m.id) for m in to_compress]
        return {"conversation_summary": new_summary, "messages": removals}

    return summarize

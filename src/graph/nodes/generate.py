"""Grounded answer generation against retrieved context."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from src.graph.deps import Deps
from src.graph.prompts import (
    GENERATE_PROMPT,
    GENERATE_SYSTEM,
    INSUFFICIENT_CONTEXT,
    sanitize_untrusted,
)
from src.graph.state import VoicebotState
from src.rag.store import SearchResult
from src.utils.config import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_context(docs: list[SearchResult]) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, 1):
        text = sanitize_untrusted(d.text_en)
        if d.headings:
            # Prepend the section-heading breadcrumb so the model has the
            # structural context the chunk came from (esp. for tables).
            parts.append(f"[{i}] ({sanitize_untrusted(d.headings)})\n{text}")
        else:
            parts.append(f"[{i}] {text}")
    return "\n\n---\n\n".join(parts)


# Native Ollama thinking returns reasoning in a separate field, but strip any
# inline <think>/<|think|> blocks defensively so chain-of-thought never reaches
# the translator/TTS as part of the answer.
_THINK_TAGS = re.compile(r"<\|?think\|?>.*?</\|?think\|?>", re.DOTALL | re.IGNORECASE)
# A response truncated mid-thought (e.g. hit max tokens) leaves an opening
# tag with no closing one - _THINK_TAGS won't match it, so drop everything
# from that point on rather than leak raw chain-of-thought into TTS.
_THINK_OPEN_UNCLOSED = re.compile(r"<\|?think\|?>.*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    text = _THINK_TAGS.sub("", text)
    text = _THINK_OPEN_UNCLOSED.sub("", text)
    return text.strip()


def make_generate(deps: Deps) -> Callable[[VoicebotState], dict[str, Any]]:
    def generate(state: VoicebotState) -> dict[str, Any]:
        docs: list[SearchResult] = state.get("retrieved_docs") or []
        if not docs:
            # Never ask the LLM to answer from an empty context - it will invent.
            return {
                "english_response": "",
                "fallback_triggered": True,
                "fallback_reason": "no_context",
            }

        query = state.get("rewritten_query") or state.get("english_query", "")
        context = _build_context(docs)

        try:
            resp = deps.llm.complete(
                messages=[
                    {"role": "system", "content": GENERATE_SYSTEM},
                    {
                        "role": "user",
                        "content": GENERATE_PROMPT.format(context=context, query=query),
                    },
                ],
                model=cfg["llm"]["model"],
                think=bool(cfg["llm"].get("think_on_generate", False)),
                temperature=0.05,  # low: grounded QA, minimal invention
            )
        except Exception:
            logger.exception("Generation LLM failed - degrading to fallback")
            return {
                "english_response": "",
                "fallback_triggered": True,
                "fallback_reason": "llm_error",
            }
        answer = _strip_thinking(resp.content)

        # Prefix match, not substring: context is sanitized so the sentinel can
        # only come from the model itself, and instructed output starts with it.
        if answer.startswith(INSUFFICIENT_CONTEXT):
            return {
                "english_response": "",
                "fallback_triggered": True,
                "fallback_reason": "insufficient_context",
            }
        return {"english_response": answer}

    return generate

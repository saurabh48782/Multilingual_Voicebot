"""Second-pass LLM check: is the generated answer fully supported by context?

Runs only when generate produced a non-empty answer. Failure routes to fallback.
This is a safety guard, so it fails CLOSED: a verifier crash or unparseable
response is treated as ungrounded rather than waving the answer through.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.utils.json import parse_json_markdown
from pydantic import BaseModel

from src.graph.deps import Deps
from src.graph.nodes.generate import _build_context
from src.graph.prompts import (
    GROUNDEDNESS_PROMPT,
    GROUNDEDNESS_SYSTEM,
    sanitize_untrusted,
)
from src.graph.state import VoicebotState
from src.utils.config import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


_THINK_TAGS = re.compile(r"<\|?think\|?>.*?</\|?think\|?>", re.DOTALL | re.IGNORECASE)


class GroundednessVerdict(BaseModel):
    grounded: bool
    reasoning: str = ""


def _strip_thinking(content: str) -> str:
    return _THINK_TAGS.sub("", content).strip()


def _parse_json_object(content: str) -> Any:
    """Parse a JSON object even if the model wrapped it in extra text."""
    content = _strip_thinking(content)
    try:
        return parse_json_markdown(content, parser=json.loads)
    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        for index, char in enumerate(content):
            if char != "{":
                continue
            try:
                data, _ = decoder.raw_decode(content[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                candidates.append(data)

        if candidates:
            # If the model rambles before the answer, the final valid object is
            # most likely the actual verdict rather than an example/schema echo.
            return candidates[-1]
        raise original_error


def _preview(content: Any, limit: int = 240) -> str:
    text = "" if content is None else str(content)
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def parse_verdict(content: str) -> bool:
    """Parse the verifier's JSON verdict. Raises on anything malformed -
    callers must treat that as ungrounded (fail closed).
    """
    data = _parse_json_object(content)
    return bool(GroundednessVerdict.model_validate(data).grounded)


def make_verify_groundedness(deps: Deps) -> Callable[[VoicebotState], dict[str, Any]]:
    def verify(state: VoicebotState) -> dict[str, Any]:
        if state.get("fallback_triggered"):
            return {"grounded": False}

        answer = state.get("english_response", "")
        if not answer:
            return {"grounded": False}

        query = state.get("rewritten_query") or state.get("english_query", "")
        context = _build_context(state.get("retrieved_docs") or [])
        reason = "ungrounded"
        try:
            resp = deps.llm.complete(
                messages=[
                    {"role": "system", "content": GROUNDEDNESS_SYSTEM},
                    {
                        "role": "user",
                        "content": GROUNDEDNESS_PROMPT.format(
                            context=context,
                            query=sanitize_untrusted(query),
                            answer=sanitize_untrusted(answer),
                        ),
                    },
                ],
                # A different-family model is used to avoid correlated
                # failure modes between generator and verifier.
                model=cfg["llm"].get("verifier_model"),
                json_mode=True,
                temperature=0.0,  # deterministic safety gate
                num_ctx=cfg["llm"].get("verifier_num_ctx") or cfg["llm"].get("num_ctx"),
            )
        except Exception:
            logger.exception("Groundedness verifier call failed - failing closed")
            grounded = False
            reason = "verifier_error"
        else:
            try:
                grounded = parse_verdict(resp.content)
            except Exception as exc:
                logger.warning(
                    "Groundedness verifier returned unparseable verdict - failing closed",
                    error=f"{type(exc).__name__}: {exc}",
                    response_preview=_preview(resp.content),
                )
                grounded = False
                reason = "verifier_error"

        if not grounded:
            return {
                "grounded": False,
                "fallback_triggered": True,
                "fallback_reason": reason,
            }
        return {"grounded": True}

    return verify


def route_after_verify(state: VoicebotState) -> str:
    return "fallback" if state.get("fallback_triggered") else "translate_to_vernacular"

"""Second-pass LLM check: is the generated answer fully supported by context?

Runs only when generate produced a non-empty answer. Failure routes to fallback.
This is a safety guard, so it fails CLOSED: a verifier crash or unparseable
response is treated as ungrounded rather than waving the answer through.
"""

from __future__ import annotations

import json
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


class GroundednessVerdict(BaseModel):  # type: ignore[misc]
    grounded: bool
    reasoning: str = ""


def parse_verdict(content: str) -> bool:
    """Parse the verifier's JSON verdict. Raises on anything malformed -
    callers must treat that as ungrounded (fail closed).
    """
    data = parse_json_markdown(content.strip(), parser=json.loads)
    return GroundednessVerdict.model_validate(data).grounded  # type: ignore[no-any-return]


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
                # Defaults to the shared generation model, but a different-family
                # model can be pinned via llm.verifier_model to avoid correlated
                # failure modes between generator and verifier.
                model=cfg["llm"].get("verifier_model") or cfg["llm"]["model"],
                json_mode=True,
                temperature=0.0,  # deterministic safety gate
            )
            grounded = parse_verdict(resp.content)
        except Exception:
            logger.exception("Groundedness verifier failed - failing closed")
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

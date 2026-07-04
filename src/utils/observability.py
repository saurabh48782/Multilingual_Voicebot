"""LangSmith tracing setup + helpers.

Tracing is opt-in. When ``observability.langsmith.enabled`` is false (default),
``configure_tracing`` leaves it off and ``@traceable`` is a near-zero-cost
pass-through (the SDK checks the env at call time). ``configure_tracing`` maps our
config onto the ``LANGSMITH_*`` env vars the SDK reads; it is idempotent so every
entrypoint (API lifespan, CLIs) can call it safely.
"""

from __future__ import annotations

import os
from typing import Any

from langsmith import traceable

from src.utils.config import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ``traceable`` is re-exported for the non-Runnable clients (LLM/STT/TTS).
__all__ = [
    "traceable",
    "configure_tracing",
    "tracing_enabled",
    "graph_run_config",
    "strip_self",
    "llm_run_outputs",
    "redact_audio_inputs",
    "redact_audio_outputs",
]


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def configure_tracing() -> bool:
    """Map ``observability.langsmith`` config onto the ``LANGSMITH_*`` env vars the
    SDK reads (it honours the legacy ``LANGCHAIN_*`` names too, so one namespace is
    enough). Idempotent. Returns whether tracing ended up on."""

    ls = (cfg.get("observability") or {}).get("langsmith") or {}
    enabled = _truthy(ls.get("enabled", False))
    api_key = str(ls.get("api_key") or "").strip()

    if not enabled or not api_key:
        # Force-off so a stray LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 in the
        # environment can't enable tracing against a half-configured project.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        if enabled:
            logger.warning(
                "observability.langsmith.enabled is true but no api_key "
                "(LANGSMITH_API_KEY) is set - tracing stays OFF."
            )
        else:
            logger.info("langsmith tracing disabled")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = str(ls.get("project") or "voicebot").strip()
    os.environ["LANGSMITH_ENDPOINT"] = str(
        ls.get("endpoint") or "https://api.smith.langchain.com"
    ).strip()

    logger.info(
        "langsmith tracing enabled",
        project=os.environ["LANGSMITH_PROJECT"],
        endpoint=os.environ["LANGSMITH_ENDPOINT"],
    )
    return True


def tracing_enabled() -> bool:
    """True when tracing is currently on (matches what ``configure_tracing`` wrote)."""

    return _truthy(os.environ.get("LANGSMITH_TRACING")) or _truthy(
        os.environ.get("LANGCHAIN_TRACING_V2")
    )


def graph_run_config(
    session_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the LangGraph ``RunnableConfig`` for one turn.

    ``thread_id`` drives the checkpointer; ``run_name``/``tags``/``metadata``
    make each turn a named, filterable trace in LangSmith (searchable by
    session_id / language). None-valued metadata entries are dropped."""
    meta: dict[str, Any] = {"session_id": session_id}
    if metadata:
        meta.update({k: v for k, v in metadata.items() if v is not None})
    return {
        "configurable": {"thread_id": session_id},
        "run_name": "voicebot_turn",
        "tags": ["voicebot"],
        "metadata": meta,
    }


# @traceable process_inputs / process_outputs helpers
def strip_self(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in inputs.items() if k != "self"}


def llm_run_outputs(resp: Any) -> dict[str, Any]:
    """Shape an ``LLMResponse`` so LangSmith records token usage for the span."""
    usage = getattr(resp, "usage", None) or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    return {
        "content": getattr(resp, "content", ""),
        "model": getattr(resp, "model", ""),
        "usage_metadata": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def redact_audio_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    cleaned = strip_self(inputs)
    audio = cleaned.get("audio")
    if isinstance(audio, bytes | bytearray):
        cleaned["audio"] = {"audio_bytes": len(audio)}
    return cleaned


def redact_audio_outputs(out: Any) -> dict[str, Any]:
    if isinstance(out, bytes | bytearray):
        return {"audio_bytes": len(out)}
    return {"output": out}

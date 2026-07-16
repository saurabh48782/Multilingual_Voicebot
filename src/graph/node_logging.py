"""Per-node / per-route structured tracing for the voicebot graph.

`build_graph` wraps every node with `traced_node` and every conditional-edge
router with `traced_router`, so each super-step of a turn emits:

    node.start   node=<name>
    node.done    node=<name> ms=<float> <compact summary of what changed>
    route        node=<name> to=<branch>

and node failures log `node.error` before re-raising. `run_graph` binds
`session_id` / `input_mode` / `language` onto the log contextvars for the turn,
so every one of these lines carries them and a single session can be filtered
out of the terminal or the JSON log file with `jq`.

Nothing large or sensitive is logged: audio bytes, retrieved docs and message
objects are reduced to counts, and long free-text fields to a short preview.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from src.graph.state import VoicebotState
from src.utils.logger import get_logger

logger = get_logger("graph")

_PREVIEW = 160

# Scalar state keys worth echoing verbatim on node.done.
_SCALAR_KEYS = (
    "input_mode",
    "source_language",
    "intent",
    "retrieval_confidence",
    "retrieval_gap",
    "retrieval_passed",
    "grounded",
    "fallback_triggered",
    "fallback_reason",
    "audio_id",
    "audio_content_type",
)

# Long free-text keys → logged as a truncated single-line preview.
_TEXT_KEYS = (
    "transcript",
    "english_query",
    "rewritten_query",
    "english_response",
    "vernacular_response",
    "conversation_summary",
)


def _preview(value: str) -> str:
    value = value.replace("\n", " ").strip()
    return value if len(value) <= _PREVIEW else value[: _PREVIEW - 1] + "…"


def _summarize(update: dict[str, Any]) -> dict[str, Any]:
    """Reduce a node's partial-state update to a compact, safe log payload."""
    out: dict[str, Any] = {}
    for key in _SCALAR_KEYS:
        if key in update and update[key] not in (None, ""):
            out[key] = update[key]
    for key in _TEXT_KEYS:
        value = update.get(key)
        if isinstance(value, str) and value:
            out[key] = _preview(value)
    if "retrieved_docs" in update:
        out["retrieved_docs"] = len(update.get("retrieved_docs") or [])
    if "messages" in update:
        out["messages_added"] = len(update.get("messages") or [])
    return out


def traced_node(
    name: str,
    fn: Callable[[VoicebotState], dict[str, Any]],
) -> Callable[[VoicebotState], dict[str, Any]]:
    """Wrap a graph node so it logs entry, exit (+ duration + summary) and errors."""

    def wrapped(state: VoicebotState) -> dict[str, Any]:
        logger.info("node.start", node=name)
        start = time.perf_counter()
        try:
            update = fn(state)
        except Exception:
            ms = round((time.perf_counter() - start) * 1000, 1)
            logger.exception("node.error", node=name, ms=ms)
            raise
        ms = round((time.perf_counter() - start) * 1000, 1)
        payload = _summarize(update) if isinstance(update, dict) else {}
        logger.info("node.done", node=name, ms=ms, **payload)
        return update

    wrapped.__name__ = name
    return wrapped


def traced_router(
    name: str,
    fn: Callable[[VoicebotState], str],
) -> Callable[[VoicebotState], str]:
    """Wrap a conditional-edge router so it logs the branch it selected."""

    def wrapped(state: VoicebotState) -> str:
        decision = fn(state)
        logger.info("route", node=name, to=decision)
        return decision

    wrapped.__name__ = name
    return wrapped

"""Helpers that bridge the LangGraph state dict ↔ public Pydantic schemas.

Kept separate from routers so both /voice and /chat reuse identical logic.
All graph access goes through the async LangGraph API (`ainvoke`,
`aget_state`, `adelete_thread`) because production compiles the graph with
`AsyncPostgresSaver`, whose sync methods are not usable; LangGraph runs the
sync node functions in worker threads, so blocking transformer calls never
stall the event loop.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import structlog
from langchain_core.messages import AIMessage, HumanMessage

from src.api.schemas import (
    ConfidenceReport,
    ContextChunk,
    SessionHistory,
    SessionHistoryEntry,
    VoiceResponse,
)
from src.db.session_store import touch_session
from src.rag.store import SearchResult
from src.utils.logger import get_logger
from src.utils.observability import graph_run_config

logger = get_logger("graph")


def _overall_band(
    retrieval_passed: bool, grounded: bool, fallback: bool
) -> Literal["green", "yellow", "red"]:
    if fallback or not retrieval_passed:
        return "red"
    if not grounded:
        return "yellow"
    return "green"


def state_to_response(
    state: dict[str, Any],
    session_id: str,
) -> VoiceResponse:
    docs: list[SearchResult] = state.get("retrieved_docs") or []
    chunks = [
        ContextChunk(
            text=d.text_en,
            score=d.score,
            source=d.source,
            page_num=d.page_num,
        )
        for d in docs
    ]

    audio_id = state.get("audio_id")
    audio_content_type = state.get("audio_content_type")
    audio_url: str | None = None
    if audio_id:
        ext = "mp3" if audio_content_type == "audio/mpeg" else "wav"
        audio_url = f"/audio/{audio_id}.{ext}"

    fallback_triggered = bool(state.get("fallback_triggered"))
    retrieval_passed = bool(state.get("retrieval_passed"))
    grounded = bool(state.get("grounded"))

    return VoiceResponse(
        session_id=session_id,
        transcript=state.get("transcript", ""),
        detected_language=state.get("source_language", "en"),
        english_query=state.get("english_query", ""),
        rewritten_query=state.get("rewritten_query") or state.get("english_query", ""),
        retrieved_context=chunks,
        response_text=state.get("vernacular_response", ""),
        response_text_english=state.get("english_response", ""),
        audio_url=audio_url,
        audio_content_type=audio_content_type,
        confidence=ConfidenceReport(
            retrieval_score=float(state.get("retrieval_confidence") or 0.0),
            retrieval_gap=float(state.get("retrieval_gap") or 0.0),
            retrieval_passed=retrieval_passed,
            grounded=grounded,
            overall=_overall_band(retrieval_passed, grounded, fallback_triggered),
        ),
        fallback_triggered=fallback_triggered,
        fallback_reason=state.get("fallback_reason") or None,
    )


async def run_graph(
    graph: Any,
    inputs: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    """Execute one graph turn. Returns the merged state dict.

    The run config names/tags the turn so it lands in LangSmith as a
    filterable trace (session_id, language, voice vs text). session_id /
    input_mode / language are also bound onto the log contextvars for the turn,
    so every per-node log line (see src/graph/node_logging.py) carries them."""
    input_mode = "voice" if inputs.get("audio_input") else "text"
    language = inputs.get("requested_language")
    cfg = graph_run_config(
        session_id,
        metadata={"language": language, "input_mode": input_mode},
    )

    structlog.contextvars.bind_contextvars(
        session_id=session_id,
        input_mode=input_mode,
        language=language,
    )
    logger.info(
        "turn.start",
        input_mode=input_mode,
        has_audio=bool(inputs.get("audio_input")),
        chars=len(inputs.get("text_input") or ""),
    )
    start = time.perf_counter()
    try:
        state: dict[str, Any] = await graph.ainvoke(inputs, config=cfg)
        logger.info(
            "turn.done",
            ms=round((time.perf_counter() - start) * 1000, 1),
            intent=state.get("intent"),
            retrieval_passed=state.get("retrieval_passed"),
            grounded=state.get("grounded"),
            fallback_triggered=bool(state.get("fallback_triggered")),
            fallback_reason=state.get("fallback_reason") or None,
            audio_id=state.get("audio_id"),
        )
        return state
    except Exception:
        logger.exception("turn.error", ms=round((time.perf_counter() - start) * 1000, 1))
        raise
    finally:
        structlog.contextvars.unbind_contextvars("session_id", "input_mode", "language")


async def record_turn(
    pool: Any,
    session_id: str,
    user_text: str,
    state: dict[str, Any],
) -> None:
    """Refresh session metadata after a completed turn. ``user_text`` seeds the
    title on the first turn; the turn count is read back from the checkpointed
    ``messages`` so it stays in sync with what's actually persisted."""
    message_count = len(state.get("messages") or [])
    await touch_session(pool, session_id, title_hint=user_text, message_count=message_count)


async def history_from_state(graph: Any, session_id: str) -> SessionHistory:
    """Read the latest checkpointed `messages` list for a session."""
    cfg = {"configurable": {"thread_id": session_id}}
    snapshot = await graph.aget_state(cfg)
    messages = (snapshot.values or {}).get("messages") or []
    entries: list[SessionHistoryEntry] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            entries.append(SessionHistoryEntry(role="user", content=str(msg.content)))
        elif isinstance(msg, AIMessage):
            entries.append(SessionHistoryEntry(role="assistant", content=str(msg.content)))
    return SessionHistory(session_id=session_id, messages=entries)


async def reset_session(checkpointer: Any, session_id: str) -> None:
    """Drop all checkpoints for a thread_id."""
    if hasattr(checkpointer, "adelete_thread"):
        await checkpointer.adelete_thread(session_id)
    elif hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(session_id)
    else:
        # fallback for in-memory checkpointers (used in tests)
        store = getattr(checkpointer, "storage", None)
        if isinstance(store, dict):
            store.pop(session_id, None)


__all__ = [
    "history_from_state",
    "record_turn",
    "reset_session",
    "run_graph",
    "state_to_response",
]

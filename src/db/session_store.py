"""Best-effort session metadata store: titles, timestamps, turn counts.
Conversation *content* lives in the LangGraph Postgres checkpointer (keyed by
``thread_id``); this table only holds the lightweight metadata the frontend
needs to *list* and *resume* past chats.
"""

from __future__ import annotations

from typing import Any

from pypika import Case, Order, Parameter, PostgreSQLQuery, Table
from pypika.terms import Function

from src.utils.logger import get_logger

logger = get_logger(__name__)

chat = Table("chat_sessions")


def make_title(text: str) -> str:
    """Derive a sidebar title from the first user utterance: collapse
    whitespace and truncate. Vernacular text is kept as-is (the first user
    turn is stored in the original language)."""
    max_len = 60
    title = " ".join((text or "").split())
    if len(title) > max_len:
        title = title[: max_len - 1].rstrip() + "…"
    return title


async def touch_session(
    pool: Any,
    session_id: str,
    *,
    title_hint: str,
    message_count: int,
) -> None:
    """Upsert metadata after a turn. ``title`` is set once (on the first turn
    that produces a non-empty hint) and never overwritten; ``last_active`` and
    ``message_count`` always refresh. Best-effort - failures are logged."""
    if pool is None:
        return
    query = (
        PostgreSQLQuery.into(chat)
        .columns(chat.session_id, chat.title, chat.last_active, chat.message_count)
        .insert(Parameter("$1"), Parameter("$2"), Function("NOW"), Parameter("$3"))
        .on_conflict(chat.session_id)
        .do_update(chat.last_active, Function("NOW"))
        .do_update(chat.message_count, Parameter("$3"))
        .do_update(
            chat.title,
            Case().when(chat.title == "", Parameter("$2")).else_(chat.title),
        )
    )
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                str(query),
                session_id,
                make_title(title_hint),
                message_count,
            )
    except Exception:
        logger.exception("failed to record session metadata", session_id=session_id)


async def list_sessions(pool: Any) -> list[dict[str, Any]]:
    """All known sessions, most-recently-active first."""
    if pool is None:
        return []

    query = (
        PostgreSQLQuery.from_(chat)
        .select(
            chat.session_id,
            chat.title,
            chat.created_at,
            chat.last_active,
            chat.message_count,
        )
        .orderby(chat.last_active, order=Order.desc)
    )
    # Best-effort, like touch_session: a metadata read failure must not 500 the
    # sidebar - the checkpointer (conversation content) is the source of truth.
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(str(query))
    except Exception:
        logger.exception("failed to list session metadata")
        return []
    return [dict(r) for r in rows]


async def delete_session(pool: Any, session_id: str) -> None:
    """Drop a session's metadata row (checkpoints are cleared separately)."""
    if pool is None:
        return
    query = PostgreSQLQuery.from_(chat).where(chat.session_id == Parameter("$1")).delete()
    try:
        async with pool.acquire() as conn:
            await conn.execute(str(query), session_id)
    except Exception:
        logger.exception("failed to delete session metadata", session_id=session_id)


__all__ = [
    "delete_session",
    "list_sessions",
    "make_title",
    "touch_session",
]

"""Session listing, inspection, reset, and delete endpoints.

`messages` content note: the checkpointed history stores the assistant turn's
`content` in English (for the coreference rewrite - see `synthesize.py`), but
the vernacular reply the user saw is preserved in the message's
`additional_kwargs["vernacular"]`. `history_from_state` prefers that vernacular
copy, so replayed history renders in the original language.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from src.api.deps import CheckpointerDep, DbPoolDep, GraphDep, SessionLocksDep
from src.api.schemas import SessionHistory, SessionList, SessionMeta
from src.api.services import history_from_state, reset_session
from src.api.session_locks import SessionLocks
from src.db import session_store

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


@router.get(
    "",
    response_model=SessionList,
    summary="List all sessions",
    description="Retrieve metadata for all saved chat sessions (title, timestamps, turn count)",
)
async def list_sessions(pool: Any = DbPoolDep) -> SessionList:
    rows = await session_store.list_sessions(pool)
    return SessionList(sessions=[SessionMeta(**row) for row in rows])


@router.get(
    "/{session_id}",
    response_model=SessionHistory,
    summary="Get session history",
    description="Retrieve full conversation history for a session "
    "(messages with user queries in vernacular and bot replies)",
)
async def get_session(
    session_id: str,
    graph: Any = GraphDep,
) -> SessionHistory:
    return await history_from_state(graph, session_id)


@router.post(
    "/{session_id}/reset",
    summary="Clear session messages",
    description="Clear the conversation history while keeping "
    "the session in the list with reset turn count",
)
async def reset(
    session_id: str,
    checkpointer: Any = CheckpointerDep,
    pool: Any = DbPoolDep,
    locks: SessionLocks = SessionLocksDep,
) -> dict[str, str]:
    """Clear the conversation but keep the session listed (turn count reset)."""
    async with locks.get(session_id):
        await reset_session(checkpointer, session_id)
        await session_store.touch_session(pool, session_id, title_hint="", message_count=0)
    return {"session_id": session_id, "status": "cleared"}


@router.delete(
    "/{session_id}",
    summary="Delete session",
    description="Permanently delete a session: removes checkpoints and metadata (cannot be undone)",
)
async def delete_session(
    session_id: str,
    checkpointer: Any = CheckpointerDep,
    pool: Any = DbPoolDep,
    locks: SessionLocks = SessionLocksDep,
) -> dict[str, str]:
    """Remove the session entirely: drop its checkpoints and metadata row."""
    async with locks.get(session_id):
        await reset_session(checkpointer, session_id)
        await session_store.delete_session(pool, session_id)
    return {"session_id": session_id, "status": "deleted"}

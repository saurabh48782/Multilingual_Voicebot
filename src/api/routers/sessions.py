"""Session listing, inspection, reset, and delete endpoints.

`messages` content note: the checkpointed history stores the user turn in the
original (vernacular) language but the assistant turn in English - see
`synthesize.py`, which keeps assistant text in English for the coreference
rewrite. Replayed history therefore shows vernacular questions with English
answers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from src.api.deps import CheckpointerDep, DbPoolDep, GraphDep, SessionLocksDep
from src.api.schemas import SessionHistory, SessionList, SessionMeta
from src.api.services import history_from_state, reset_session
from src.api.session_locks import SessionLocks
from src.db import session_store

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=SessionList)
async def list_sessions(pool: Any = DbPoolDep) -> SessionList:
    rows = await session_store.list_sessions(pool)
    return SessionList(sessions=[SessionMeta(**row) for row in rows])


@router.get("/{session_id}", response_model=SessionHistory)
async def get_session(
    session_id: str,
    graph: Any = GraphDep,
) -> SessionHistory:
    return await history_from_state(graph, session_id)


@router.post("/{session_id}/reset")
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


@router.delete("/{session_id}")
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

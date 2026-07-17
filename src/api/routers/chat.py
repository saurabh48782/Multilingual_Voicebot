"""POST /api/chat - text input → end-to-end vernacular reply."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from src.api.deps import DbPoolDep, GraphDep, SessionLocksDep
from src.api.schemas import ChatRequest, VoiceResponse
from src.api.services import record_turn, run_graph, state_to_response
from src.api.session_locks import SessionLocks

router = APIRouter(prefix="/api", tags=["Chat"])


@router.post(
    "/chat",
    response_model=VoiceResponse,
    summary="Chat with voice bot",
    description="Send a text query and get a response from the voice bot",
)
async def chat(
    payload: ChatRequest,
    graph: Any = GraphDep,
    pool: Any = DbPoolDep,
    locks: SessionLocks = SessionLocksDep,
) -> VoiceResponse:
    inputs: dict[str, Any] = {
        "text_input": payload.text,
        "requested_language": payload.language,
    }
    async with locks.get(payload.session_id):
        state = await run_graph(graph, inputs, session_id=payload.session_id)
        await record_turn(pool, payload.session_id, payload.text, state)
    return state_to_response(state, session_id=payload.session_id)

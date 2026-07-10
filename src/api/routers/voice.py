"""POST /api/voice - multipart audio → end-to-end vernacular voice reply."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.deps import DbPoolDep, GraphDep, SessionLocksDep
from src.api.schemas import VoiceResponse
from src.api.services import record_turn, run_graph, state_to_response
from src.api.session_locks import SessionLocks
from src.translation.base import SUPPORTED_VERNACULARS

router = APIRouter(prefix="/api", tags=["voice"])

_MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB ≈ several minutes of 16-bit WAV


@router.post("/voice", response_model=VoiceResponse)
async def voice(
    session_id: Annotated[str, Form(min_length=1)],
    audio: Annotated[UploadFile, File()],
    # IndicConformer does not auto-detect language, so the caller must
    # supply the spoken language (the UI mandates a selection).
    language: Annotated[str, Form(min_length=1)],
    graph: Any = GraphDep,
    pool: Any = DbPoolDep,
    locks: SessionLocks = SessionLocksDep,
) -> VoiceResponse:
    # Voice is restricted to the vernaculars the STT model handles. IndicConformer is
    # Indic-only (English is not one of its languages), so `en` audio cannot be transcribed.
    if language not in SUPPORTED_VERNACULARS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"voice language must be one of {sorted(SUPPORTED_VERNACULARS)}; "
                "English speech is not supported by the STT model (use the text tab for English)"
            ),
        )
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    if len(raw) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio exceeds 25 MB limit")

    inputs: dict[str, Any] = {
        "audio_input": raw,
        "requested_language": language,
    }
    async with locks.get(session_id):
        state = await run_graph(graph, inputs, session_id=session_id)
        title_hint = state.get("transcript", "")
        await record_turn(pool, session_id, title_hint, state)
    return state_to_response(state, session_id=session_id)

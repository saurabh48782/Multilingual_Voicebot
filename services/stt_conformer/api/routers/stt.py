"""POST /stt - multipart audio → IndicConformer transcription.

multipart: audio=<file>, language=<code>, decode_strategy=rnnt|ctc. The blocking
model call runs in a threadpool so the event loop stays free. IndicConformer does
not auto-detect language - the caller must pass it (the app's UI mandates it).
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from api.deps import TranscriberDep
from api.schemas import TranscriptionResponse
from api.transcriber import Transcriber

router = APIRouter(tags=["stt"])

# Mirrors the main app's per-request cap (src/api/routers/voice.py) - this
# sidecar has no outer body-size middleware of its own.
_MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB
_VALID_DECODE_STRATEGIES = {"rnnt", "ctc"}

# Module-level singletons: FastAPI dependency markers must not be constructed in
# argument defaults (ruff B008).
_AUDIO_FILE = File(...)
_LANGUAGE_FORM = Form(...)
_DECODE_STRATEGY_FORM = Form("rnnt")


@router.post("/stt", response_model=TranscriptionResponse)
async def stt(
    audio: UploadFile = _AUDIO_FILE,
    language: str = _LANGUAGE_FORM,
    decode_strategy: str = _DECODE_STRATEGY_FORM,
    transcriber: Transcriber = TranscriberDep,
) -> TranscriptionResponse:
    strategy = decode_strategy.lower()
    if strategy not in _VALID_DECODE_STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail=f"decode_strategy must be one of {sorted(_VALID_DECODE_STRATEGIES)}",
        )
    raw = await audio.read()
    if len(raw) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio exceeds 25 MB limit")
    lang = language.lower()
    text = await run_in_threadpool(transcriber.transcribe, raw, lang, strategy)
    return TranscriptionResponse(text=text, language=lang)

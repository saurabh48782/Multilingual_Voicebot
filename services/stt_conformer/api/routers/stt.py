"""POST /stt - multipart audio → IndicConformer transcription.

multipart: audio=<file>, language=<code>, decode_strategy=rnnt|ctc. The blocking
model call runs in a threadpool so the event loop stays free. IndicConformer does
not auto-detect language - the caller must pass it (the app's UI mandates it).
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from api.deps import TranscriberDep
from api.schemas import TranscriptionResponse
from api.transcriber import Transcriber

router = APIRouter(tags=["stt"])


@router.post("/stt", response_model=TranscriptionResponse)
async def stt(
    audio: UploadFile = File(...),
    language: str = Form(...),
    decode_strategy: str = Form("rnnt"),
    transcriber: Transcriber = TranscriberDep,
) -> TranscriptionResponse:
    raw = await audio.read()
    lang = language.lower()
    text = await run_in_threadpool(
        transcriber.transcribe, raw, lang, decode_strategy.lower()
    )
    return TranscriptionResponse(text=text, language=lang)

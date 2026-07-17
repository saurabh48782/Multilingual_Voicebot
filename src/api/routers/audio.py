"""GET /audio/{audio_id}.{ext} - stream cached TTS bytes."""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.api.audio_cache import AudioCache
from src.api.deps import AudioCacheDep

router = APIRouter(prefix="/audio", tags=["Audio"])

_VALID_ID = re.compile(r"^[a-f0-9]{8,64}$")


@router.get(
    "/{audio_filename}",
    summary="Stream cached audio",
    description="Retrieve cached TTS (text-to-speech) audio files by ID (WAV or MP3)",
)
def stream_audio(
    audio_filename: str,
    audio_cache: AudioCache = AudioCacheDep,
) -> FileResponse:
    audio_id, _, ext = audio_filename.partition(".")
    if not _VALID_ID.match(audio_id) or ext not in {"wav", "mp3"}:
        raise HTTPException(status_code=400, detail="invalid audio id")

    path = audio_cache.path_for(audio_id)
    if path is None:
        raise HTTPException(status_code=404, detail="audio not found or expired")

    media_type = "audio/mpeg" if path.suffix == ".mp3" else "audio/wav"
    return FileResponse(path, media_type=media_type)

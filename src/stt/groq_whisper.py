"""Groq Whisper-large-v3 STT (primary path).

Groq's verbose_json response includes a `language` field (ISO 639-1) derived
directly from Whisper's language token - used as source_language in the graph.
"""

from __future__ import annotations

import io

from groq import Groq

from src.stt.base import TranscriptionResult
from src.utils.config import cfg

_client: Groq | None = None

# Groq Whisper returns full language names ("hindi") - map to ISO codes.
_LANG_NAME_TO_CODE: dict[str, str] = {
    "hindi": "hi",
    "bengali": "bn",
    "english": "en",
}


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = cfg["api"]["groq_api_key"]
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set - add it to .env")
        # Longer timeout than the chat client: audio uploads are bigger payloads.
        _client = Groq(api_key=key, timeout=60.0, max_retries=2)
    return _client


class GroqWhisperSTT:
    """Calls Groq audio.transcriptions.create with whisper-large-v3."""

    def transcribe(
        self,
        audio: bytes,
        language: str | None = None,
    ) -> TranscriptionResult:
        resp = _get_client().audio.transcriptions.create(
            file=("audio.wav", io.BytesIO(audio), "audio/wav"),
            model=cfg["stt"]["whisper_model"],
            response_format="verbose_json",
            **({"language": language} if language else {}),  # type: ignore[arg-type]
        )

        raw_lang: str = getattr(resp, "language", "") or language or "en"
        iso_lang = _LANG_NAME_TO_CODE.get(raw_lang.lower(), raw_lang.lower()[:2])

        return TranscriptionResult(
            text=(resp.text or "").strip(),
            language=iso_lang,
        )

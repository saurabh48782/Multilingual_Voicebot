"""STT providers: Groq Whisper (primary) + faster-whisper local (fallback)."""

from src.stt.base import STTProvider, TranscriptionResult
from src.stt.faster_whisper_local import FasterWhisperSTT
from src.stt.groq_whisper import GroqWhisperSTT

__all__ = ["STTProvider", "TranscriptionResult", "GroqWhisperSTT", "FasterWhisperSTT"]

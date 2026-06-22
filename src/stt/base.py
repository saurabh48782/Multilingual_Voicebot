"""STT provider protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class TranscriptionResult:
    text: str
    language: str  # ISO 639-1: "hi" | "bn" | "en"
    confidence: float = 1.0


@runtime_checkable
class STTProvider(Protocol):
    def transcribe(
        self,
        audio: bytes,
        language: str | None = None,
    ) -> TranscriptionResult: ...

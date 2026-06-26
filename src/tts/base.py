"""TTS provider protocol.

synthesize() returns raw audio bytes.
  - Indic Parler-TTS: WAV (PCM 16-bit, model sampling rate)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TTSProvider(Protocol):
    def synthesize(self, text: str, language: str) -> bytes:
        """Return audio bytes for text in the given language.

        Args:
            text: Text to synthesize.
            language: ISO 639-1 code ("hi" | "bn" | "en").
        """
        ...

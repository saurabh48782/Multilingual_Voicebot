"""faster-whisper local STT (GPU fallback / offline mode).

Model is loaded lazily on first transcribe call.
Device selection: CUDA if torch reports it available, else CPU with int8.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from src.stt.base import TranscriptionResult
from src.utils.config import cfg


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class FasterWhisperSTT:
    """Local faster-whisper inference."""

    def __init__(self) -> None:
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        device = "cuda" if _has_cuda() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(
            cfg["stt"]["faster_whisper_model"],
            device=device,
            compute_type=compute_type,
        )

    def transcribe(
        self,
        audio: bytes,
        language: str | None = None,
    ) -> TranscriptionResult:
        self._load()

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                tmp_path = f.name

            kwargs: dict[str, str] = {}
            if language:
                kwargs["language"] = language

            segments, info = self._model.transcribe(tmp_path, **kwargs)
            text = " ".join(s.text for s in segments).strip()
            return TranscriptionResult(
                text=text,
                language=info.language or language or "en",
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

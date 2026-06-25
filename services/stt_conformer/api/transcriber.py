"""AI4Bharat IndicConformer model loader + transcription.

The 600M model loads lazily on the first transcribe() call and stays resident, so
the container reports healthy immediately and the load cost is paid once.
"""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Any

import torch
import torchaudio
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
TARGET_SR = 16000


class Transcriber:
    """Lazy, thread-safe singleton wrapper around the IndicConformer model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self._device: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(
                device
            )
            model.eval()
            self._model, self._device = model, device

    def _load_waveform(self, audio: bytes) -> Any:
        with tempfile.NamedTemporaryFile(suffix="", delete=False) as f:
            f.write(audio)
            path = f.name
        try:
            wav, sr = torchaudio.load(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != TARGET_SR:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
        return wav

    def transcribe(self, audio: bytes, language: str, decode_strategy: str) -> str:
        self._ensure_loaded()
        wav = self._load_waveform(audio).to(self._device)
        with torch.no_grad():
            out = self._model(wav, language, decode_strategy)
        if isinstance(out, str):
            return out.strip()
        if isinstance(out, (list, tuple)):
            return " ".join(str(x) for x in out).strip()
        return str(out).strip()

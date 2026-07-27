"""AI4Bharat IndicConformer model loader + transcription.

The 600M model loads lazily on the first transcribe() call into page-locked CPU
RAM, so the container reports healthy immediately and the load cost is paid once.
It is paged onto the GPU only for the duration of each transcribe() call and
evicted immediately afterwards (its VRAM handed back to the driver), so the GPU
is free for the co-hosted Ollama LLM / TTS sidecar between calls. See _GpuSwap.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torchaudio
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
TARGET_SR = 16000


class _GpuSwap:
    """Keep a model's weights page-locked in CPU RAM and page them onto the GPU
    only for the span of a request, freeing the VRAM again immediately after.

    No-op when running CPU-only (no CUDA). The caller must hold the model lock
    around ``on_gpu()`` so two requests can't page the model in/out concurrently.
    """

    def __init__(self, model: Any, device: str) -> None:
        self._model = model
        self._device = device
        self._enabled = torch.cuda.is_available() and str(device) != "cpu"
        self._cpu_params: dict[str, torch.Tensor] = {}
        self._cpu_buffers: dict[str, torch.Tensor] = {}
        if self._enabled:
            # named_parameters()/named_buffers() dedupe shared storage, so tied
            # weights are pinned + restored once and stay tied.
            for name, p in model.named_parameters():
                p.data = p.data.pin_memory()
                self._cpu_params[name] = p.data
            for name, b in model.named_buffers():
                if b.device.type == "cpu":
                    b.data = b.data.pin_memory()
                self._cpu_buffers[name] = b.data

    @contextmanager
    def on_gpu(self) -> Iterator[None]:
        if not self._enabled:
            yield
            return
        self._model.to(self._device, non_blocking=True)
        try:
            yield
        finally:
            for name, p in self._model.named_parameters():
                p.data = self._cpu_params[name]
            for name, b in self._model.named_buffers():
                cpu = self._cpu_buffers.get(name)
                if cpu is not None:
                    b.data = cpu
            torch.cuda.empty_cache()


class Transcriber:
    """Lazy, thread-safe singleton wrapper around the IndicConformer model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any = None
        self._device: str | None = None
        self._swap: _GpuSwap | None = None

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
            # Load onto CPU; _GpuSwap pins the weights and pages them to the GPU
            # per request rather than holding VRAM for the container's lifetime.
            model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
            model.eval()
            self._model, self._device = model, device
            self._swap = _GpuSwap(model, device)

    @staticmethod
    def _decode_with_ffmpeg(path: str) -> Any:
        """Decode any container/codec ffmpeg understands to 16 kHz mono float32.

        Needed because torchaudio only registers the `soundfile` backend here
        (libsndfile reads wav/flac/ogg but not webm/opus or mp4/aac, which is
        what browser MediaRecorder produces), and torchaudio 2.4's own ffmpeg
        backend refuses the image's ffmpeg 7 (it supports 4-6 only). The ffmpeg
        CLI is installed in the image, so shell out to it instead.
        """
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            msg = "ffmpeg not found in PATH"
            raise RuntimeError(msg)
        proc = subprocess.run(  # noqa: S603
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                path,
                "-f",
                "f32le",  # raw little-endian float32 samples on stdout
                "-ac",
                "1",  # downmix to mono
                "-ar",
                str(TARGET_SR),  # resample to the model's rate
                "-",
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            detail = proc.stderr.decode("utf-8", "replace").strip()[-500:]
            msg = f"ffmpeg could not decode the uploaded audio: {detail}"
            raise ValueError(msg)
        samples = torch.frombuffer(bytearray(proc.stdout), dtype=torch.float32)
        return samples.unsqueeze(0)  # (1, num_samples)

    def _load_waveform(self, audio: bytes) -> Any:
        with tempfile.NamedTemporaryFile(suffix="", delete=False) as f:
            f.write(audio)
            path = f.name
        try:
            try:
                wav, sr = torchaudio.load(path)
            except Exception:
                # Compressed browser upload (webm/opus, mp4/aac, ...) - libsndfile
                # can't read it. ffmpeg already returns mono at TARGET_SR.
                return self._decode_with_ffmpeg(path)
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
        wav = self._load_waveform(audio)
        assert self._swap is not None  # set alongside self._model in _ensure_loaded  # noqa: S101
        # Serialize inference AND scope GPU residency to this call: under the lock
        # the model is paged onto the GPU, used, then evicted. Concurrent requests
        # sharing one GPU-resident model can each allocate enough memory to OOM it,
        # so the lock guards both concerns.
        with self._lock, self._swap.on_gpu(), torch.no_grad():
            wav = wav.to(self._device)
            out = self._model(wav, language, decode_strategy)
            if torch.is_tensor(out):
                out = out.cpu()
        if isinstance(out, str):
            return out.strip()
        if isinstance(out, list | tuple):
            return " ".join(str(x) for x in out).strip()
        return str(out).strip()

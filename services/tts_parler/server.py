"""Indic Parler-TTS sidecar service.

Endpoints:
  GET  /healthz      → {"status": "ok", "model_loaded": bool}
  POST /tts          → audio/wav bytes   body: {"text": str, "description": str?}

The 0.9B model loads lazily on the first /tts request into page-locked CPU RAM.
It is paged onto the GPU only for the duration of each /tts call and evicted
immediately afterwards (its VRAM handed back to the driver), so the GPU is free
for the co-hosted Ollama LLM / STT sidecar between synthesis calls. See _GpuSwap.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from parler_tts import ParlerTTSForConditionalGeneration
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

MODEL_ID = "ai4bharat/indic-parler-tts"
_MAX_TEXT_LEN = 2000  # caps generation cost from an unbounded input string
DEFAULT_DESCRIPTION = (
    "A clear, calm voice speaks at a natural, moderate pace in a quiet "
    "environment. The recording is very high quality, with the voice sounding "
    "close-up and natural, with no background noise."
)

app = FastAPI(title="indic-parler-tts")

_lock = threading.Lock()
_state: dict[str, Any] = {
    "model": None,
    "prompt_tok": None,
    "desc_tok": None,
    "device": None,
    "swap": None,
}


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


def _ensure_loaded() -> None:
    if _state["model"] is not None:
        return
    with _lock:
        if _state["model"] is not None:
            return
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Load onto CPU; _GpuSwap pins the weights and pages them to the GPU
        # per request rather than holding VRAM for the container's lifetime.
        model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID)
        model.eval()
        _state.update(
            model=model,
            prompt_tok=AutoTokenizer.from_pretrained(MODEL_ID),
            desc_tok=AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path),
            device=device,
            swap=_GpuSwap(model, device),
        )


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=_MAX_TEXT_LEN)
    description: str | None = None


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/tts")
def tts(req: TtsRequest) -> Response:
    _ensure_loaded()
    model, prompt_tok, desc_tok, device, swap = (
        _state["model"],
        _state["prompt_tok"],
        _state["desc_tok"],
        _state["device"],
        _state["swap"],
    )
    description = req.description or DEFAULT_DESCRIPTION

    # Serialize generate() calls AND scope GPU residency to this call: under the
    # lock the model is paged onto the GPU, used, then evicted. Concurrent
    # requests sharing one GPU-resident model can each allocate enough activation
    # memory to OOM the device, so the lock guards both concerns.
    with _lock, swap.on_gpu(), torch.no_grad():
        desc_ids = desc_tok(description, return_tensors="pt").to(device)
        prompt_ids = prompt_tok(req.text, return_tensors="pt").to(device)
        generation = model.generate(
            input_ids=desc_ids.input_ids,
            attention_mask=desc_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
        )
        # Pull the waveform to host memory before the model is evicted below.
        wav: np.ndarray = generation.cpu().numpy().squeeze()

    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=model.config.sampling_rate, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # noqa: S104

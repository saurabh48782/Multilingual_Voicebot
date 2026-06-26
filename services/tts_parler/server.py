"""Indic Parler-TTS sidecar service.

Endpoints:
  GET  /healthz      → {"status": "ok", "model_loaded": bool}
  POST /tts          → audio/wav bytes   body: {"text": str, "description": str?}

The 0.9B model loads lazily on the first /tts request and stays resident, so the
container reports healthy immediately and synthesis cost is paid once.
"""

from __future__ import annotations

import io
import threading
from typing import Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from parler_tts import ParlerTTSForConditionalGeneration
from pydantic import BaseModel
from transformers import AutoTokenizer

MODEL_ID = "ai4bharat/indic-parler-tts"
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
}


def _ensure_loaded() -> None:
    if _state["model"] is not None:
        return
    with _lock:
        if _state["model"] is not None:
            return
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
        model.eval()
        _state.update(
            model=model,
            prompt_tok=AutoTokenizer.from_pretrained(MODEL_ID),
            desc_tok=AutoTokenizer.from_pretrained(
                model.config.text_encoder._name_or_path
            ),
            device=device,
        )


class TtsRequest(BaseModel):
    text: str
    description: str | None = None


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/tts")
def tts(req: TtsRequest) -> Response:
    _ensure_loaded()
    model, prompt_tok, desc_tok, device = (
        _state["model"],
        _state["prompt_tok"],
        _state["desc_tok"],
        _state["device"],
    )
    description = req.description or DEFAULT_DESCRIPTION

    desc_ids = desc_tok(description, return_tensors="pt").to(device)
    prompt_ids = prompt_tok(req.text, return_tensors="pt").to(device)

    with torch.no_grad():
        generation = model.generate(
            input_ids=desc_ids.input_ids,
            attention_mask=desc_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
        )

    wav: np.ndarray = generation.cpu().numpy().squeeze()
    buf = io.BytesIO()
    sf.write(
        buf, wav, samplerate=model.config.sampling_rate, format="WAV", subtype="PCM_16"
    )
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # noqa: S104

"""Indic Parler-TTS sidecar service.

Endpoints:
  GET  /healthz      → {"status": "ok", "model_loaded": bool}
  POST /tts          → audio/wav bytes
                       body: {"text": str, "description": str?, "language": str?}

The 0.9B model loads lazily on the first /tts request into page-locked CPU RAM.
It is paged onto the GPU only for the duration of each /tts call and evicted
immediately afterwards (its VRAM handed back to the driver), so the GPU is free
for the co-hosted Ollama LLM / STT sidecar between synthesis calls. See _GpuSwap.
"""

from __future__ import annotations

import io
import logging
import re
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
from indic_numtowords import num2words
from parler_tts import ParlerTTSForConditionalGeneration
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

logger = logging.getLogger("indic-parler-tts")

MODEL_ID = "ai4bharat/indic-parler-tts"
_MAX_TEXT_LEN = 2000  # caps generation cost from an unbounded input string
DEFAULT_DESCRIPTION = (
    "A clear, calm voice speaks at a natural, moderate pace in a quiet "
    "environment. The recording is very high quality, with the voice sounding "
    "close-up and natural, with no background noise."
)

# --- Long-text handling -----------------------------------------------------
_CHUNK_TARGET_CHARS = 220  # keeps each generate() comfortably inside its sweet spot
_CHUNK_HARD_CHARS = 320  # a single sentence longer than this gets force-split
_JOIN_SILENCE_S = 0.18  # inter-chunk pause, so the seams read as sentence breaks

# Sentence terminators: Latin .!? plus the Devanagari/Bengali danda and double
# danda, which is what actually ends a sentence in Hindi and Bengali text.
_SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+|\n+")
# Fallback split points for a single over-long sentence, in preference order.
_CLAUSE_END = re.compile(r"(?<=[,;:;])\s+")

# --- Acronym glosses --------------------------------------------------------
# Letter names are matched explicitly rather than guessed at, so ordinary
# parentheticals ("(2015)", "(चार हजार रुपये)") are left alone. Vowel-sign
# variants are permitted because the LLM transliterates inconsistently — the
# same acronym appears as both "बीबीबीपी" and "बिबिबीपी".
_DEVANAGARI_LETTERS = (
    "ए|ब[ीि]|स[ीि]|ड[ीि]|ई|एफ|ज[ीि]|एच|आई|जे|के|एल|एम|एन|ओ|प[ीि]|"
    "क्यू|आर|एस|ट[ीि]|यू|व[ीि]|डब्ल्यू|एक्स|वाई|ज़ेड|जेड"
)
_BENGALI_LETTERS = (
    "এ|ব[িী]|স[িী]|ড[িী]|ই|এফ|জ[িী]|এইচ|আই|জে|কে|এল|এম|এন|ও|প[িী]|"
    "কিউ|আর|এস|ট[িী]|ইউ|ভ[িী]|ডব্লিউ|এক্স|ওয়াই|জেড"
)
# A parenthetical made up *entirely* of two or more letter names (Indic
# transliteration or plain Latin capitals) — i.e. an acronym and nothing else.
_ACRONYM_GLOSS = re.compile(
    r"[ \t]*[(（]\s*(?:"
    rf"(?:{_DEVANAGARI_LETTERS}){{2,}}"
    rf"|(?:{_BENGALI_LETTERS}){{2,}}"
    r"|(?:[A-Z]\.?){2,}"
    r")\s*[)）]"
)
# --- Numerals ---------------------------------------------------------------
_NUM_LANGS = ("hi", "bn", "en")
_DECIMAL_WORD = {"hi": "दशमलव", "bn": "দশমিক", "en": "point"}
# Digit runs, allowing Indian-style grouping commas and one decimal part.
_NUMBER = re.compile(r"\d+(?:,\d{2,3})*(?:\.\d+)?")
# Native digit forms, normalised to ASCII before conversion.
_INDIC_DIGITS = str.maketrans("०१२३४५६७८९০১২৩৪৫৬৭৮৯", "01234567890123456789")
# Above this many digits a number is an identifier, not a quantity (account /
# phone / pincode). "two hundred million..." would be absurd, so read it out
# digit by digit instead.
_MAX_SPELLED_DIGITS = 9

# Left behind after a strip: space before punctuation, or a doubled space.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+(?=[,.;:!?।॥])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# Parler samples its voice (do_sample=True in generation_config). Without a
# fixed seed each chunk would come back in a *different* voice.
# Seeding per request makes the whole utterance one speaker and makes output reproducible.
_VOICE_SEED = 0

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


def _resolve_num_lang(language: str | None, text: str) -> str:
    """Pick the number-spelling language, falling back to the text's script."""
    if language:
        base = language.split("-")[0].lower()
        if base in _NUM_LANGS:
            return base
    if any("ঀ" <= ch <= "৿" for ch in text):
        return "bn"
    if any("ऀ" <= ch <= "ॿ" for ch in text):
        return "hi"
    return "en"


def _spell_number(token: str, lang: str) -> str:
    """Render one numeric token as words; returns it unchanged if that fails."""
    digits = token.replace(",", "")
    integer, _, fraction = digits.partition(".")
    try:
        if len(integer) > _MAX_SPELLED_DIGITS:
            # Identifier, not a quantity - read it out digit by digit.
            spoken = " ".join(num2words(int(d), lang=lang) for d in integer)
        else:
            spoken = num2words(int(integer), lang=lang)
        if fraction:
            tail = " ".join(num2words(int(d), lang=lang) for d in fraction)
            spoken = f"{spoken} {_DECIMAL_WORD[lang]} {tail}"
    except Exception:  # noqa: BLE001 - unsupported value: keep the digits
        logger.warning("number expansion failed for %r (%s)", token, lang)
        return token
    return spoken


def expand_numbers(text: str, language: str | None = None) -> str:
    """Replace digit runs with their spoken form in ``language``."""
    lang = _resolve_num_lang(language, text)
    text = text.translate(_INDIC_DIGITS)
    return _NUMBER.sub(lambda m: _spell_number(m.group(0), lang), text)


def normalize_for_speech(text: str, language: str | None = None) -> str:
    """Strip/rewrite what the TTS decoder chokes on.

    Affects audio only - the transcript shown to the user is untouched.
    """
    text = _ACRONYM_GLOSS.sub("", text)
    text = expand_numbers(text, language)
    text = _SPACE_BEFORE_PUNCT.sub("", text)
    return _MULTI_SPACE.sub(" ", text).strip()


def _force_split(sentence: str) -> list[str]:
    """Break a single over-long sentence on clause boundaries, then whitespace."""
    parts: list[str] = []
    for clause in _CLAUSE_END.split(sentence):
        if len(clause) <= _CHUNK_HARD_CHARS:
            parts.append(clause)
            continue
        # No punctuation to lean on: pack words up to the target width.
        buf = ""
        for word in clause.split():
            if buf and len(buf) + 1 + len(word) > _CHUNK_TARGET_CHARS:
                parts.append(buf)
                buf = word
            else:
                buf = f"{buf} {word}" if buf else word
        if buf:
            parts.append(buf)
    return parts


def split_text(text: str) -> list[str]:
    """Split ``text`` into sentence-aligned chunks of roughly _CHUNK_TARGET_CHARS.

    Sentences are packed greedily so that short ones share a chunk (fewer
    generate() calls) while no chunk grows long enough to trip the decoder's
    quality cliff or its ~30 s ceiling.
    """
    sentences: list[str] = []
    for raw in _SENTENCE_END.split(text.strip()):
        sentence = raw.strip()
        if not sentence:
            continue
        if len(sentence) > _CHUNK_HARD_CHARS:
            sentences.extend(p for p in (s.strip() for s in _force_split(sentence)) if p)
        else:
            sentences.append(sentence)

    chunks: list[str] = []
    buf = ""
    for sentence in sentences:
        if buf and len(buf) + 1 + len(sentence) > _CHUNK_TARGET_CHARS:
            chunks.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}" if buf else sentence
    if buf:
        chunks.append(buf)
    return chunks or [text.strip()]


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=_MAX_TEXT_LEN)
    description: str | None = None
    # Drives number expansion only (the voice comes from `description`).
    language: str | None = None


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
    chunks = split_text(normalize_for_speech(req.text, req.language))
    sample_rate = model.config.sampling_rate

    # Serialize generate() calls AND scope GPU residency to this request: under
    # the lock the model is paged onto the GPU, used, then evicted. Concurrent
    # requests sharing one GPU-resident model can each allocate enough activation
    # memory to OOM the device, so the lock guards both concerns. Every chunk is
    # synthesized inside the *same* residency window — paging per chunk would
    # cost ~200 ms each for no benefit.
    with _lock, swap.on_gpu(), torch.no_grad():
        # One voice for the whole utterance (see _VOICE_SEED).
        torch.manual_seed(_VOICE_SEED)
        desc_ids = desc_tok(description, return_tensors="pt").to(device)
        pieces: list[np.ndarray] = []
        for chunk in chunks:
            prompt_ids = prompt_tok(chunk, return_tensors="pt").to(device)
            generation = model.generate(
                input_ids=desc_ids.input_ids,
                attention_mask=desc_ids.attention_mask,
                prompt_input_ids=prompt_ids.input_ids,
                prompt_attention_mask=prompt_ids.attention_mask,
            )
            # Pull each waveform to host memory before the model is evicted below.
            pieces.append(generation.cpu().numpy().squeeze().astype(np.float32))

    if len(pieces) == 1:
        wav = pieces[0]
    else:
        pause = np.zeros(int(_JOIN_SILENCE_S * sample_rate), dtype=np.float32)
        joined: list[np.ndarray] = []
        for i, piece in enumerate(pieces):
            if i:
                joined.append(pause)
            joined.append(piece)
        wav = np.concatenate(joined)

    buf = io.BytesIO()
    sf.write(buf, wav, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # noqa: S104

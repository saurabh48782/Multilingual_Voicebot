"""Embedding wrapper for intfloat/multilingual-e5-large.

Prefixes:
  - ingestion:  "passage: <text>"
  - query time: "query: <text>"

Vectors are L2-normalised so inner product == cosine similarity.

The model is loaded onto the CPU and paged onto the GPU only for the duration
of each encode call (see `src.rag.gpu_swap`), so its ~2.5 GB of VRAM is free
for the co-hosted Ollama LLM and the TTS sidecar between retrievals.
"""

from __future__ import annotations

import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.gpu_swap import GpuSwap, resolve_device
from src.utils.config import cfg

_model: SentenceTransformer | None = None
_swap: GpuSwap | None = None
_load_lock = threading.Lock()

EMBEDDING_DIM = 1024
_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "


def _get_model() -> tuple[SentenceTransformer, GpuSwap]:
    global _model, _swap
    with _load_lock:
        if _model is None or _swap is None:
            model = SentenceTransformer(cfg["rag"]["embedding_model"], device="cpu")
            _model, _swap = model, GpuSwap(model, resolve_device())
        return _model, _swap


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Return normalised (N, 1024) float32 array for corpus passages."""
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    prefixed = [_PASSAGE_PREFIX + t for t in texts]
    model, swap = _get_model()
    with swap.on_gpu():
        vecs = model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
        )
    return vecs.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """Return normalised (1, 1024) float32 array for a single query."""
    prefixed = _QUERY_PREFIX + text
    model, swap = _get_model()
    with swap.on_gpu():
        vec = model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    return vec.astype(np.float32)

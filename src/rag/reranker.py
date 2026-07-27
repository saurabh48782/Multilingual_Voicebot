"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3 by default).

Lazy-loaded singleton. Wraps a `sentence_transformers.CrossEncoder`
and returns sigmoid-scaled relevance scores so they live in (0, 1)
and remain comparable against the existing confidence thresholds.

Weights live in pinned CPU RAM and are paged onto the GPU only for the span of
each `rerank()` call (see `src.rag.gpu_swap`), so the ~2.2 GB is free for the
co-hosted Ollama LLM and the TTS sidecar between retrievals.
"""

from __future__ import annotations

import math
import threading

from sentence_transformers import CrossEncoder

from src.rag.gpu_swap import GpuSwap, resolve_device
from src.utils.config import cfg

_model: CrossEncoder | None = None
_swap: GpuSwap | None = None
_load_lock = threading.Lock()


def _get_model() -> tuple[CrossEncoder, GpuSwap]:
    global _model, _swap
    with _load_lock:
        if _model is None or _swap is None:
            model = CrossEncoder(cfg["rag"]["reranker_model"], device="cpu")
            _model, _swap = model, GpuSwap(model, resolve_device())
        return _model, _swap


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def rerank(query: str, passages: list[str], batch_size: int = 16) -> list[float]:
    """Return one sigmoid-scaled score in (0, 1) per passage, same order."""
    if not passages:
        return []
    pairs: list[list[str]] = [[query, p] for p in passages]
    model, swap = _get_model()
    with swap.on_gpu():
        raw = model.predict(
            pairs,  # type: ignore[arg-type, unused-ignore]
            batch_size=batch_size,
            show_progress_bar=False,
        )
    return [_sigmoid(float(s)) for s in raw]

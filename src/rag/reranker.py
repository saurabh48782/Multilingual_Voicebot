"""Cross-encoder reranker (BAAI/bge-reranker-v2-m3 by default).

Lazy-loaded singleton. Wraps a `sentence_transformers.CrossEncoder`
and returns sigmoid-scaled relevance scores so they live in (0, 1)
and remain comparable against the existing confidence thresholds.
"""

from __future__ import annotations

import math

from sentence_transformers import CrossEncoder

from src.utils.config import cfg

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(cfg["rag"]["reranker_model"])
    return _model


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
    raw = _get_model().predict(
        pairs,  # type: ignore[arg-type, unused-ignore]
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return [_sigmoid(float(s)) for s in raw]

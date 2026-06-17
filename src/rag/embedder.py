"""Embedding wrapper for intfloat/multilingual-e5-large.

Prefixes:
  - ingestion:  "passage: <text>"
  - query time: "query: <text>"

Vectors are L2-normalised so inner product == cosine similarity.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import cfg

_model: SentenceTransformer | None = None

EMBEDDING_DIM = 1024
_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(cfg["rag"]["embedding_model"])
    return _model


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Return normalised (N, 1024) float32 array for corpus passages."""
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    prefixed = [_PASSAGE_PREFIX + t for t in texts]
    vecs = _get_model().encode(
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
    vec = _get_model().encode(
        [prefixed],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype(np.float32)

"""LangChain wrappers around our Ollama LLM + multilingual-e5 embedder for RAGAS.

See src/evaluation/README.md ("Judge configuration") for why each option is set.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from src.utils.config import cfg


class E5Embeddings(Embeddings):  # type: ignore[misc, unused-ignore]
    """HF E5 embedder that prepends the correct prefix per call.

    E5 needs `passage: ` on documents and `query: ` on queries; mixing them collapses
    quality. Mirrors `src/rag/embedder.py` so RAGAS shares the index's vector space.
    """

    def __init__(self, model_name: str) -> None:
        self._inner = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents([f"passage: {t}" for t in texts])  # type: ignore[no-any-return, unused-ignore]

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(f"query: {text}")  # type: ignore[no-any-return, unused-ignore]


@lru_cache(maxsize=1)
def judge_llm() -> ChatOllama:
    """Ollama-hosted Gemma judge LLM for RAGAS metrics (shares the prod Ollama server)."""
    eval_cfg = cfg["evaluation"]
    return ChatOllama(
        model=eval_cfg["judge_model"],
        base_url=cfg["llm"]["ollama_base_url"],
        temperature=float(eval_cfg.get("judge_temperature", 0.0)),
        format="json",
        num_predict=int(eval_cfg.get("judge_num_predict", 2048)),
        reasoning=bool(eval_cfg.get("judge_reasoning", False)),
        # Always set: Ollama's default window truncates from the FRONT, dropping the
        # RAGAS instructions (mirrors production llm.num_ctx).
        num_ctx=int(eval_cfg.get("judge_num_ctx", cfg["llm"].get("num_ctx", 8192))),
    )


@lru_cache(maxsize=1)
def judge_embeddings() -> E5Embeddings:
    """E5 embedder shared with the RAG index (so cosine spaces line up)."""
    return E5Embeddings(model_name=cfg["evaluation"]["embedding_model"])


def ragas_llm_wrapper() -> Any:
    """Return a RAGAS-compatible LLM wrapper around the Ollama Gemma judge."""
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(judge_llm())


def ragas_embeddings_wrapper() -> Any:
    """Return a RAGAS-compatible embeddings wrapper around the E5 embedder."""
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(judge_embeddings())

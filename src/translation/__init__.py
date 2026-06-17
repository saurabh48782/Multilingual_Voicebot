"""Translation layer: Ollama (default) or Groq LLM."""

from __future__ import annotations

from src.translation.base import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_VERNACULARS,
    TranslationResult,
)
from src.translation.base import TranslationProvider as TranslationProtocol
from src.translation.groq_llm import GroqTranslator
from src.utils.config import cfg
from src.utils.providers import TranslationProvider

__all__ = [
    "GroqTranslator",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_VERNACULARS",
    "TranslationProtocol",
    "TranslationResult",
    "get_translator",
]


def get_translator() -> TranslationProtocol:
    """Return the configured translation provider."""
    if (
        TranslationProvider(cfg["translation"]["provider"])
        is TranslationProvider.OLLAMA
    ):
        from src.translation.ollama_translator import OllamaTranslator

        return OllamaTranslator()
    return GroqTranslator()

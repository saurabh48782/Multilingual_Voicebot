"""Translation layer: Ollama."""

from __future__ import annotations

from src.translation.base import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_VERNACULARS,
    TranslationResult,
)
from src.translation.base import TranslationProvider as TranslationProtocol

__all__ = [
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_VERNACULARS",
    "TranslationProtocol",
    "TranslationResult",
    "get_translator",
]


def get_translator() -> TranslationProtocol:
    from src.translation.ollama_translator import OllamaTranslator

    return OllamaTranslator()

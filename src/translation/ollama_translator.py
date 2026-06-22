"""Ollama LLM-as-translator (privacy-preserving, no API key required).

Shares all translation logic with :class:`LLMTranslator`; this class only wires
up the local Ollama client and model.
"""

from __future__ import annotations

from src.llm.ollama_client import OllamaLLM
from src.translation.llm_translator import LLMTranslator
from src.utils.config import cfg


class OllamaTranslator(LLMTranslator):
    """Translation provider backed by a locally-hosted Ollama LLM."""

    def __init__(self, model: str | None = None) -> None:
        self._llm = OllamaLLM()
        self._model: str = model or cfg["translation"]["ollama_model"]

"""Groq LLM-as-translator.

Uses the Groq chat completions endpoint with a deterministic, instruction-only
prompt so the model emits ONLY the translation (no preamble, no quotes). The
translation logic itself lives in :class:`LLMTranslator`; this class only wires
up the Groq client and model.
"""

from __future__ import annotations

from src.llm.base import LLMProvider
from src.llm.groq_client import GroqLLM
from src.translation.llm_translator import LLMTranslator
from src.utils.config import cfg


class GroqTranslator(LLMTranslator):
    """Translation provider backed by a Groq-hosted LLM (Llama family)."""

    def __init__(
        self, llm: LLMProvider | None = None, model: str | None = None
    ) -> None:
        self._llm: LLMProvider = llm or GroqLLM()
        self._model: str = model or cfg["llm"]["model"]

"""Shared LLM-as-translator logic.

``GroqTranslator`` and ``OllamaTranslator`` differ only in which LLM client and
model they wire up; the routing, prompt construction, quote-stripping, and
batch path are identical, so they live here and are inherited. Subclasses set
``self._llm`` and ``self._model`` in their ``__init__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.translation.base import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_VERNACULARS,
    TranslationResult,
    translate_batch_to_english,
)
from src.translation.prompts import (
    LANG_NAMES,
    LANG_SCRIPTS,
    TO_EN_SYSTEM,
    TO_VERN_SYSTEM,
)

if TYPE_CHECKING:
    from src.llm.base import LLMProvider


def _clean(content: str) -> str:
    """Strip a single pair of matching outer quotes the model may have added."""
    text = content.strip()
    if len(text) >= 2 and text[0] in "\"'“‘" and text[-1] in "\"'”’":
        text = text[1:-1].strip()
    return text


class LLMTranslator:
    """Translation provider backed by an arbitrary chat-completion LLM.

    Subclasses must set ``self._llm`` (an :class:`LLMProvider`) and
    ``self._model`` before any translate call.
    """

    _llm: LLMProvider
    _model: str

    def to_english(self, text: str, source_language: str) -> TranslationResult:
        if source_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported source language: {source_language!r}")
        if source_language == "en" or not text.strip():
            return TranslationResult(
                text=text, source_language="en", target_language="en"
            )

        system = TO_EN_SYSTEM.format(src_name=LANG_NAMES[source_language])
        resp = self._llm.complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            model=self._model,
        )
        return TranslationResult(
            text=_clean(resp.content),
            source_language=source_language,
            target_language="en",
        )

    def to_vernacular(self, text: str, target_language: str) -> TranslationResult:
        if target_language not in SUPPORTED_VERNACULARS:
            if target_language == "en":
                return TranslationResult(
                    text=text, source_language="en", target_language="en"
                )
            raise ValueError(f"Unsupported target language: {target_language!r}")
        if not text.strip():
            return TranslationResult(
                text=text, source_language="en", target_language=target_language
            )

        system = TO_VERN_SYSTEM.format(
            tgt_name=LANG_NAMES[target_language],
            tgt_script=LANG_SCRIPTS[target_language],
        )
        resp = self._llm.complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            model=self._model,
        )
        return TranslationResult(
            text=_clean(resp.content),
            source_language="en",
            target_language=target_language,
        )

    def to_english_batch(self, texts: list[str]) -> list[str]:
        return translate_batch_to_english(self._llm, self._model, texts)

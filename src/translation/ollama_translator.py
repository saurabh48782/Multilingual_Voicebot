"""Ollama LLM-as-translator (privacy-preserving, no API key required)."""

from __future__ import annotations

from src.llm.ollama_client import OllamaLLM
from src.translation.base import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_VERNACULARS,
    TranslationResult,
    translate_batch_to_english,
)
from src.utils.config import cfg

LANG_NAMES: dict[str, str] = {
    "hi": "Hindi",
    "bn": "Bengali",
    "en": "English",
}

LANG_SCRIPTS: dict[str, str] = {
    "hi": "Devanagari",
    "bn": "Bengali script",
}

_TO_EN_SYSTEM = (
    "You are a professional translator. Translate the user's text from "
    "{src_name} to English. Output ONLY the English translation. "
    "Do not add quotes, explanations, transliterations, or any other text. "
    "Preserve proper nouns (scheme names, organisations) verbatim. "
    "If the input is already English, return it unchanged."
)

_TO_VERN_SYSTEM = (
    "You are a professional translator. Translate the user's text from "
    "English to {tgt_name}. Output ONLY the {tgt_name} translation in the "
    "native script ({tgt_script}). Do not add quotes, explanations, "
    "transliterations, or any other text. Preserve scheme names verbatim."
)


def _clean(content: str) -> str:
    text = content.strip()
    if len(text) >= 2 and text[0] in "\"'‘“" and text[-1] in "\"'’”":
        text = text[1:-1].strip()
    return text


class OllamaTranslator:
    """Translation provider backed by a locally-hosted Ollama LLM."""

    def __init__(self, model: str | None = None) -> None:
        self._llm = OllamaLLM()
        self._model: str = model or cfg["translation"]["ollama_model"]

    def to_english(self, text: str, source_language: str) -> TranslationResult:
        if source_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported source language: {source_language!r}")
        if source_language == "en" or not text.strip():
            return TranslationResult(
                text=text, source_language="en", target_language="en"
            )

        system = _TO_EN_SYSTEM.format(src_name=LANG_NAMES[source_language])
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

        system = _TO_VERN_SYSTEM.format(
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

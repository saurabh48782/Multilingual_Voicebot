"""Shared prompt templates and language metadata for LLM-backed translators.

The translators emit the same deterministic, instruction-only prompts
(model returns ONLY the translation, no preamble or quotes),
so the templates and language lookups live here and are reused.
"""

from __future__ import annotations

LANG_NAMES: dict[str, str] = {
    "hi": "Hindi",
    "bn": "Bengali",
    "en": "English",
}

LANG_SCRIPTS: dict[str, str] = {
    "hi": "Devanagari",
    "bn": "Bengali script",
}

TO_EN_SYSTEM = (
    "You are a professional translator. Translate the user's text from "
    "{src_name} to English. Output ONLY the English translation. "
    "Do not add quotes, explanations, transliterations, or any other text. "
    "Preserve proper nouns (scheme names, organisations) verbatim. "
    "If the input is already English, return it unchanged."
)

TO_VERN_SYSTEM = (
    "You are a professional translator. Translate the user's text from "
    "English to {tgt_name}. Output ONLY the {tgt_name} translation in the "
    "native script ({tgt_script}). Do not add quotes, explanations, "
    "transliterations, or any other text. Preserve scheme names verbatim."
)

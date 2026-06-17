"""Valid provider identifiers for runtime switching via params.yaml."""

from enum import StrEnum


class LLMProvider(StrEnum):
    GROQ = "groq"
    OLLAMA = "ollama"


class TranslationProvider(StrEnum):
    OLLAMA = "ollama"
    GROQ = "groq"


class TTSProvider(StrEnum):
    MMS = "mms"
    INDICF5 = "indicf5"
    GTTS = "gtts"

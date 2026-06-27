"""Valid provider identifiers for runtime switching via params.yaml."""

from enum import StrEnum


class LLMProvider(StrEnum):
    OLLAMA = "ollama"


class TranslationProvider(StrEnum):
    OLLAMA = "ollama"


class TTSProvider(StrEnum):
    INDIC_PARLER = "indic_parler"


class STTProvider(StrEnum):
    INDIC_CONFORMER = "indic_conformer"

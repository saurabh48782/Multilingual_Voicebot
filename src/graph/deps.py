"""Provider container - single place every node grabs its dependencies from."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.llm.base import LLMProvider as LLMProto
from src.stt.base import STTProvider
from src.translation.base import TranslationProvider as TranslationProto
from src.tts.base import TTSProvider as TTSProto
from src.utils.config import cfg
from src.utils.providers import LLMProvider, TranslationProvider, TTSProvider

if TYPE_CHECKING:
    from src.rag.retriever import Retriever


def _default_stt() -> STTProvider:
    from src.stt.groq_whisper import GroqWhisperSTT

    return GroqWhisperSTT()


def _default_tts() -> TTSProto:
    if TTSProvider(cfg["tts"]["provider"]) is TTSProvider.INDICF5:
        from src.tts.indic_f5 import IndicF5Tts

        return IndicF5Tts()

    if TTSProvider(cfg["tts"]["provider"]) is TTSProvider.GTTS:
        from src.tts.gtts_fallback import GttsFallback

        return GttsFallback()

    from src.tts.mms_tts import MMSTts

    return MMSTts()


def _default_llm() -> LLMProto:
    if LLMProvider(cfg["llm"]["provider"]) is LLMProvider.OLLAMA:
        from src.llm.ollama_client import OllamaLLM

        return OllamaLLM()
    from src.llm.groq_client import GroqLLM

    return GroqLLM()


def _default_translator() -> TranslationProto:
    provider = TranslationProvider(cfg["translation"]["provider"])
    if provider is TranslationProvider.OLLAMA:
        from src.translation.ollama_translator import OllamaTranslator

        return OllamaTranslator()
    from src.translation.groq_llm import GroqTranslator

    return GroqTranslator()


def _default_retriever() -> Retriever:
    from src.rag.retriever import Retriever

    return Retriever.load()


@dataclass
class Deps:
    stt: STTProvider = field(default_factory=_default_stt)
    tts: TTSProto = field(default_factory=_default_tts)
    translator: TranslationProto = field(default_factory=_default_translator)
    llm: LLMProto = field(default_factory=_default_llm)
    retriever: Retriever = field(default_factory=_default_retriever)

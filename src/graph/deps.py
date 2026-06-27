"""Provider container - single place every node grabs its dependencies from."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.llm.base import LLMProvider as LLMProto
from src.stt.base import STTProvider as STTProto
from src.translation.base import TranslationProvider as TranslationProto
from src.tts.base import TTSProvider as TTSProto
from src.utils.config import cfg
from src.utils.providers import STTProvider, TTSProvider

if TYPE_CHECKING:
    from src.rag.retriever import Retriever


def _default_stt() -> STTProto:
    # It runs only as the services/stt_conformer sidecar,
    # so stt.remote_url is required.
    STTProvider(cfg["stt"]["provider"])  # validate config value
    remote_url = (cfg["stt"].get("remote_url") or "").strip()
    if not remote_url:
        raise RuntimeError(
            "stt.remote_url (env STT_REMOTE_URL) is not set - start the "
            "services/stt_conformer sidecar (docker compose sets http://stt:8002)."
        )
    from src.stt.indic_conformer_remote import IndicConformerRemoteStt

    return IndicConformerRemoteStt(remote_url)


def _default_tts() -> TTSProto:
    # It runs only as the services/tts_parler sidecar,
    # so tts.remote_url is required.
    TTSProvider(cfg["tts"]["provider"])  # validate config value
    remote_url = (cfg["tts"].get("remote_url") or "").strip()
    if not remote_url:
        raise RuntimeError(
            "tts.remote_url (env TTS_REMOTE_URL) is not set - start the "
            "services/tts_parler sidecar (docker compose sets http://tts:8001)."
        )
    from src.tts.indic_parler_remote import IndicParlerRemoteTts

    return IndicParlerRemoteTts(remote_url)


def _default_llm() -> LLMProto:
    from src.llm.ollama_client import OllamaLLM

    return OllamaLLM()


def _default_translator() -> TranslationProto:
    from src.translation.ollama_translator import OllamaTranslator

    return OllamaTranslator()


def _default_retriever() -> Retriever:
    from src.rag.retriever import Retriever

    return Retriever.load()


@dataclass
class Deps:
    stt: STTProto = field(default_factory=_default_stt)
    tts: TTSProto = field(default_factory=_default_tts)
    translator: TranslationProto = field(default_factory=_default_translator)
    llm: LLMProto = field(default_factory=_default_llm)
    retriever: Retriever = field(default_factory=_default_retriever)

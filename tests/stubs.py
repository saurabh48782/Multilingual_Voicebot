"""Shared deterministic stub providers for graph/API tests.

Single source of truth for the fake STT/translator/LLM/retriever/TTS/audio
cache used across unit and integration tests, so provider-interface drift
(e.g. a new ``complete()`` kwarg) only needs fixing in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.llm.base import LLMResponse
from src.rag.retriever import RetrievalResult
from src.rag.store import SearchResult
from src.stt.base import TranscriptionResult
from src.translation.base import TranslationResult


@dataclass
class StubSTT:
    text: str = "पीएम किसान योजना क्या है?"
    language: str = "hi"
    calls: list[bytes] = field(default_factory=list)

    def transcribe(self, audio: bytes, language: str | None = None) -> TranscriptionResult:
        self.calls.append(audio)
        return TranscriptionResult(text=self.text, language=self.language)


@dataclass
class StubTranslator:
    english: str = "What is the PM Kisan scheme?"
    vernacular: str = "पीएम किसान योजना का जवाब।"
    to_english_calls: list[tuple[str, str]] = field(default_factory=list)
    to_vern_calls: list[tuple[str, str]] = field(default_factory=list)

    def to_english(self, text: str, source_language: str) -> TranslationResult:
        self.to_english_calls.append((text, source_language))
        return TranslationResult(
            text=self.english, source_language=source_language, target_language="en"
        )

    def to_vernacular(self, text: str, target_language: str) -> TranslationResult:
        self.to_vern_calls.append((text, target_language))
        return TranslationResult(
            text=self.vernacular, source_language="en", target_language=target_language
        )

    def to_english_batch(self, texts: list[str]) -> list[str]:
        return [self.english for _ in texts]


@dataclass
class StubLLM:
    """Scriptable LLM. Order: rewrite (if history) -> generate -> groundedness."""

    generate_response: str = "PM Kisan provides ₹6000/year direct income support to farmers."
    groundedness_response: str = '{"grounded": true, "reasoning": "ok"}'
    rewrite_response: str = "What is the PM Kisan scheme follow-up?"
    # Default SCHEME keeps existing tests flowing through the full RAG path.
    classify_response: str = "SCHEME"
    smalltalk_response: str = "Hello! Ask me about a government scheme."
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        json_mode: bool = False,
        think: bool = False,
        temperature: float | None = None,
        num_ctx: int | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        self.calls.append(
            {"system": messages[0]["content"], "user": messages[-1]["content"], "json": json_mode}
        )
        if json_mode:
            return LLMResponse(content=self.groundedness_response, model="stub")
        system = messages[0]["content"]
        if "intent classifier" in system:
            return LLMResponse(content=self.classify_response, model="stub")
        if "small talk" in system:
            return LLMResponse(content=self.smalltalk_response, model="stub")
        if "Rewritten query" in messages[-1]["content"]:
            return LLMResponse(content=self.rewrite_response, model="stub")
        return LLMResponse(content=self.generate_response, model="stub")


@dataclass
class StubRetriever:
    passed: bool = True
    top_score: float = 0.9
    gap: float = 0.12
    docs: list[SearchResult] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.docs:
            self.docs = [
                SearchResult(
                    chunk_id="abc",
                    doc_id="pmkisan",
                    chunk_index=0,
                    text_en="PM Kisan provides ₹6000/year.",
                    source="pm_kisan.pdf",
                    page_num=1,
                    score=self.top_score,
                ),
            ]

    def search(self, query: str, k: int | None = None) -> RetrievalResult:
        self.queries.append(query)
        return RetrievalResult(
            docs=self.docs, top_score=self.top_score, gap=self.gap, passed=self.passed
        )


@dataclass
class StubTTS:
    calls: list[tuple[str, str]] = field(default_factory=list)

    def synthesize(self, text: str, language: str) -> bytes:
        self.calls.append((text, language))
        return b"RIFF\x00\x00\x00\x00WAVE" + text.encode("utf-8")[:16]


@dataclass
class StubAudioCache:
    """In-memory stand-in - avoids hitting real disk in graph-routing tests."""

    puts: list[bytes] = field(default_factory=list)

    def put(self, audio: bytes, content_type: str = "audio/wav") -> str:
        self.puts.append(audio)
        return f"stub-audio-{len(self.puts)}"

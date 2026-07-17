"""LangGraph routing tests - all external clients mocked.

Covers:
  (a) Hindi audio happy path: stt → translate → retrieve(pass) → generate
      → groundedness(pass) → translate_to_vernacular → synthesize → END.
  (b) Low retrieval confidence → fallback short-circuit.
  (c) Groundedness false → fallback.
  (d) Follow-up question with seeded `messages` history → rewriter produces a
      self-contained query.
  (e) PII in the transcript is scrubbed before the LLM sees it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_graph
from src.graph.deps import Deps
from src.llm.base import LLMResponse
from src.stt.base import TranscriptionResult
from src.translation.base import TranslationResult
from tests.stubs import StubAudioCache, StubLLM, StubRetriever, StubSTT, StubTranslator, StubTTS


# helpers
def _make_deps(**overrides: Any) -> tuple[Deps, dict[str, Any]]:
    stubs: dict[str, Any] = {
        "stt": StubSTT(),
        "tts": StubTTS(),
        # This suite's assertions check the literal fallback-vs-answer text,
        # so pin a distinctive sentinel instead of the shared module default.
        "translator": StubTranslator(vernacular="VERN_RESPONSE"),
        "llm": StubLLM(),
        "retriever": StubRetriever(),
        "audio_cache": StubAudioCache(),
    }
    stubs.update(overrides)
    deps = Deps(
        stt=stubs["stt"],
        tts=stubs["tts"],
        translator=stubs["translator"],
        llm=stubs["llm"],
        retriever=stubs["retriever"],
        audio_cache=stubs["audio_cache"],
    )
    return deps, stubs


def _invoke(deps: Deps, state: dict[str, Any], thread_id: str = "t1") -> dict[str, Any]:
    graph = build_graph(checkpointer=MemorySaver(), deps=deps)
    result: dict[str, Any] = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
    return result


# tests
def test_happy_path_hindi_audio() -> None:
    deps, stubs = _make_deps()
    result = _invoke(deps, {"audio_input": b"AUDIOBYTES", "requested_language": "hi"})

    assert stubs["stt"].calls == [b"AUDIOBYTES"]
    assert result["transcript"] == "पीएम किसान योजना क्या है?"
    assert result["source_language"] == "hi"
    assert result["english_query"] == "What is the PM Kisan scheme?"
    assert result["rewritten_query"] == "What is the PM Kisan scheme?"  # no history → passthrough
    assert result["retrieval_passed"] is True
    assert result["english_response"].startswith("PM Kisan")
    assert result["grounded"] is True
    assert result["vernacular_response"] == "VERN_RESPONSE"
    assert result["audio_id"]
    assert result["audio_content_type"] == "audio/wav"
    assert not result.get("fallback_triggered")
    # Memory turn appended
    messages = result.get("messages") or []
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)


def test_general_query_skips_rag() -> None:
    """A greeting is answered by smalltalk without touching retrieve/generate/verify."""
    deps, stubs = _make_deps(
        llm=StubLLM(classify_response="GENERAL", smalltalk_response="Hi Saurabh! Ask me anything."),
    )
    result = _invoke(deps, {"text_input": "Hello my name is Saurabh", "requested_language": "en"})

    # RAG pipeline was bypassed entirely.
    assert stubs["retriever"].queries == []
    assert not any(c["json"] for c in stubs["llm"].calls)  # no groundedness verify
    assert all(
        not c["system"].startswith("You are a careful") for c in stubs["llm"].calls
    )  # no gen
    # Direct small-talk answer, translated but NOT synthesized (text-in turn).
    assert result["intent"] == "general"
    assert result["english_response"] == "Hi Saurabh! Ask me anything."
    # en source: translate_to_vernacular passes the English answer through unchanged.
    assert result["vernacular_response"] == "Hi Saurabh! Ask me anything."
    # Text input → text-only reply: no TTS, no audio.
    assert result["audio_id"] is None
    assert not result.get("fallback_triggered")
    # Turn still recorded to memory.
    messages = result.get("messages") or []
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)


def test_scheme_query_still_retrieves() -> None:
    """A scheme question classifies as SCHEME (default) and flows through retrieval."""
    deps, stubs = _make_deps()
    result = _invoke(
        deps, {"text_input": "What is the PM Kisan scheme?", "requested_language": "en"}
    )

    assert result.get("intent") == "scheme"
    assert stubs["retriever"].queries  # retrieval ran
    assert result["english_response"].startswith("PM Kisan")
    assert not result.get("fallback_triggered")


def test_low_retrieval_score_routes_to_fallback() -> None:
    deps, stubs = _make_deps(retriever=StubRetriever(passed=False, top_score=0.3, gap=0.01))
    result = _invoke(deps, {"audio_input": b"AUDIOBYTES", "requested_language": "hi"})

    assert result["fallback_triggered"] is True
    assert result["fallback_reason"] == "retrieval_gate"
    # Only the intent classifier ran; generator + groundedness did NOT.
    assert stubs["llm"].calls
    assert all(c["system"].startswith("You are an intent classifier") for c in stubs["llm"].calls)
    # Fallback message is the canned Hindi text
    assert "जानकारी" in result["vernacular_response"]
    # TTS still runs on the fallback message
    assert stubs["tts"].calls and stubs["tts"].calls[0][1] == "hi"


def test_groundedness_false_routes_to_fallback() -> None:
    deps, stubs = _make_deps(
        llm=StubLLM(groundedness_response='{"grounded": false, "reasoning": "fabricated"}')
    )
    result = _invoke(deps, {"audio_input": b"AUDIOBYTES", "requested_language": "hi"})

    assert result["fallback_triggered"] is True
    assert result["fallback_reason"] == "ungrounded"
    assert result["grounded"] is False
    # Translator MUST NOT be invoked for the English answer (we route to fallback first)
    assert stubs["translator"].to_vern_calls == []
    # Fallback message in Hindi
    assert "जानकारी" in result["vernacular_response"]


def test_generator_insufficient_context_routes_to_fallback() -> None:
    deps, _ = _make_deps(llm=StubLLM(generate_response="INSUFFICIENT_CONTEXT"))
    result = _invoke(deps, {"audio_input": b"AUDIOBYTES", "requested_language": "hi"})

    assert result["fallback_triggered"] is True
    assert result["fallback_reason"] == "insufficient_context"


def test_followup_with_seeded_history_invokes_rewriter() -> None:
    """A pre-existing `messages` list should trigger the rewriter."""
    deps, stubs = _make_deps()
    seeded_messages = [
        HumanMessage(content="Tell me about PM Kisan."),
        AIMessage(content="PM Kisan gives ₹6000/year direct income support to farmers."),
    ]
    # Simulate that translation produces the follow-up's English form
    stubs["translator"].english = "Does it include loan waiver?"
    stubs["stt"].text = "इसमें लोन माफी मिलती है क्या?"

    result = _invoke(
        deps,
        {"audio_input": b"AUDIOFOLLOWUP", "messages": seeded_messages, "requested_language": "hi"},
    )

    # Rewriter must have been called (one LLM call before generate + groundedness)
    rewrite_calls = [c for c in stubs["llm"].calls if "Rewritten query" in c["user"]]
    assert rewrite_calls, "rewriter should have been invoked with non-empty history"
    assert result["rewritten_query"] == "What is the PM Kisan scheme follow-up?"
    # The retriever must search the *rewritten* query, not the raw English
    assert stubs["retriever"].queries == ["What is the PM Kisan scheme follow-up?"]


def test_pii_is_scrubbed_before_translation() -> None:
    deps, stubs = _make_deps(
        stt=StubSTT(text="My Aadhaar is 1234 5678 9012 and phone is 9876543210", language="en"),
    )
    _invoke(deps, {"audio_input": b"AUDIOBYTES", "requested_language": "hi"})

    # Translator receives scrubbed text, not raw PII
    seen = stubs["translator"].to_english_calls[0][0]
    assert "1234 5678 9012" not in seen
    assert "9876543210" not in seen
    assert "[AADHAAR]" in seen
    assert "[PHONE]" in seen


def test_multi_turn_state_persists_across_invokes() -> None:
    """Same thread_id sees the prior turn's messages on the second invoke."""
    deps, stubs = _make_deps()
    graph = build_graph(checkpointer=MemorySaver(), deps=deps)
    cfg = {"configurable": {"thread_id": "multi-turn"}}

    graph.invoke({"audio_input": b"FIRST", "requested_language": "hi"}, config=cfg)

    # Reset rewrite-detection bookkeeping
    rewrite_calls_before = sum(1 for c in stubs["llm"].calls if "Rewritten query" in c["user"])
    graph.invoke({"audio_input": b"SECOND", "requested_language": "hi"}, config=cfg)
    rewrite_calls_after = sum(1 for c in stubs["llm"].calls if "Rewritten query" in c["user"])

    assert (
        rewrite_calls_after - rewrite_calls_before == 1
    ), "second turn must trigger the rewriter because messages history is non-empty"


def test_text_input_skips_stt() -> None:
    deps, stubs = _make_deps()
    result = _invoke(
        deps,
        {"text_input": "What is PM Kisan?", "requested_language": "en"},
    )

    assert stubs["stt"].calls == []
    assert result["source_language"] == "en"
    # English-only path: translation should be no-op (passthrough)
    assert result["english_query"] == "What is the PM Kisan scheme?"  # stub returns canned


# failure-path tests
@dataclass
class PassthroughTranslator:
    """Mirrors real translator behaviour: empty/English text passes through."""

    def to_english(self, text: str, source_language: str) -> TranslationResult:
        return TranslationResult(text=text, source_language=source_language, target_language="en")

    def to_vernacular(self, text: str, target_language: str) -> TranslationResult:
        return TranslationResult(text=text, source_language="en", target_language=target_language)


@dataclass
class FailingSTT:
    def transcribe(self, audio: bytes, language: str | None = None) -> TranscriptionResult:
        raise RuntimeError("stt backend is down")


@dataclass
class FailingTTS:
    def synthesize(self, text: str, language: str) -> bytes:
        raise RuntimeError("tts model OOM")


@dataclass
class FailingGenerateLLM:
    """Raises on plain completions (generate), unreachable for json calls."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        json_mode: bool = False,
        think: bool = False,
        **_: object,  # absorb temperature/num_ctx and future kwargs
    ) -> LLMResponse:
        raise RuntimeError("llm unavailable")


def test_stt_failure_degrades_to_fallback() -> None:
    deps, _ = _make_deps(stt=FailingSTT(), translator=PassthroughTranslator())
    result = _invoke(deps, {"audio_input": b"AUDIOBYTES", "requested_language": "hi"})

    assert result["fallback_triggered"] is True
    assert result["fallback_reason"] == "stt_error"
    assert result["vernacular_response"]  # canned message still produced


def test_empty_input_routes_to_fallback_without_llm_calls() -> None:
    deps, stubs = _make_deps(translator=PassthroughTranslator())
    result = _invoke(deps, {"text_input": "", "requested_language": "en"})

    assert result["fallback_triggered"] is True
    assert result["fallback_reason"] == "empty_query"
    assert stubs["llm"].calls == []
    assert stubs["retriever"].queries == []


def test_generate_llm_failure_degrades_to_fallback() -> None:
    deps, _ = _make_deps(llm=FailingGenerateLLM())
    result = _invoke(deps, {"audio_input": b"AUDIOBYTES", "requested_language": "hi"})

    assert result["fallback_triggered"] is True
    assert result["fallback_reason"] == "llm_error"
    assert "जानकारी" in result["vernacular_response"]


def test_verifier_garbage_fails_closed() -> None:
    deps, _ = _make_deps(llm=StubLLM(groundedness_response="absolutely not json {"))
    result = _invoke(deps, {"audio_input": b"AUDIOBYTES", "requested_language": "hi"})

    assert result["fallback_triggered"] is True
    assert result["fallback_reason"] == "verifier_error"
    assert result["grounded"] is False


def test_verifier_handles_markdown_fenced_json() -> None:
    fenced = '```json\n{"grounded": true, "reasoning": "ok"}\n```'
    deps, _ = _make_deps(llm=StubLLM(groundedness_response=fenced))
    result = _invoke(deps, {"audio_input": b"AUDIOBYTES", "requested_language": "hi"})

    assert result["grounded"] is True
    assert not result.get("fallback_triggered")


def test_tts_failure_returns_text_only_response() -> None:
    deps, _ = _make_deps(tts=FailingTTS())
    result = _invoke(deps, {"audio_input": b"AUDIOBYTES", "requested_language": "hi"})

    assert result["audio_id"] is None
    assert result["vernacular_response"] == "VERN_RESPONSE"
    assert not result.get("fallback_triggered")

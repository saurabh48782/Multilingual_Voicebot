"""Unit tests for the translation layer.

Covers:
  - OllamaTranslator routing, prompt construction, quote-stripping
  - Round-trip semantic-preservation checks (5 sentences each in hi/bn)
    via a deterministic FakeLLM that emulates a translator
  - Empty-input + English-passthrough fast paths
  - Factory returns OllamaTranslator
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from src.llm.base import LLMResponse
from src.translation import get_translator
from src.translation.base import SUPPORTED_VERNACULARS
from src.translation.base import TranslationProvider as TranslationProtocol
from src.translation.ollama_translator import OllamaTranslator

# fixtures
# (vernacular, english) pairs - kept short so semantic-preservation checks are
# easy to assert. Five sentences each for hi/bn.
PAIRS: dict[str, list[tuple[str, str]]] = {
    "hi": [
        ("पीएम किसान योजना क्या है?", "What is the PM Kisan scheme?"),
        ("मुझे लोन कैसे मिलेगा?", "How will I get a loan?"),
        ("फसल बीमा के लिए आवेदन करें।", "Apply for crop insurance."),
        ("आधार कार्ड अनिवार्य है।", "Aadhaar card is mandatory."),
        ("किसान को कितनी राशि मिलती है?", "How much amount does the farmer get?"),
    ],
    "bn": [
        ("পিএম কিষাণ যোজনা কী?", "What is the PM Kisan scheme?"),
        ("আমি কীভাবে ঋণ পাব?", "How will I get a loan?"),
        ("ফসল বীমার জন্য আবেদন করুন।", "Apply for crop insurance."),
        ("আধার কার্ড বাধ্যতামূলক।", "Aadhaar card is mandatory."),
        ("কৃষক কত টাকা পান?", "How much money does the farmer get?"),
    ],
}


@dataclass
class _Call:
    messages: list[dict[str, str]]
    model: str | None
    json_mode: bool


class FakeLLM:
    """Deterministic LLM stub backed by a lookup table.

    The lookup is built from the (vernacular, english) PAIRS so that calls
    requesting "Hindi→English" return the canonical English sentence, and
    "English→Hindi" return the canonical Hindi sentence. Anything not in the
    table is echoed back so non-table assertions still work.
    """

    def __init__(self, table: dict[tuple[str, str], str]) -> None:
        self.table = table
        self.calls: list[_Call] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        json_mode: bool = False,
        think: bool = False,
        temperature: float | None = None,
        num_ctx: int | None = None,
        **_: object,
    ) -> LLMResponse:
        self.calls.append(_Call(messages=messages, model=model, json_mode=json_mode))
        system = messages[0]["content"]
        user_text = messages[-1]["content"]

        if "to English" in system:
            tgt = "en"
        else:
            # Pick whichever vernacular name appears in the system prompt.
            tgt = next(
                (code for code, name in {"hi": "Hindi", "bn": "Bengali"}.items() if name in system),
                "en",
            )
        translated = self.table.get((user_text, tgt), user_text)
        return LLMResponse(content=translated, model=model or "fake", usage={})


def _build_table() -> dict[tuple[str, str], str]:
    table: dict[tuple[str, str], str] = {}
    for lang, pairs in PAIRS.items():
        for vern, en in pairs:
            table[(vern, "en")] = en
            table[(en, lang)] = vern
    return table


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(_build_table())


@pytest.fixture
def translator(fake_llm: FakeLLM) -> OllamaTranslator:
    return OllamaTranslator(llm=fake_llm)


# protocol conformance
def test_ollama_translator_satisfies_protocol(translator: OllamaTranslator) -> None:
    assert isinstance(translator, TranslationProtocol)


# round-trip semantic checks (5 sentences per language)
@pytest.mark.parametrize("language", sorted(SUPPORTED_VERNACULARS))
def test_roundtrip_preserves_meaning(language: str, translator: OllamaTranslator) -> None:
    for vern, english in PAIRS[language]:
        en_result = translator.to_english(vern, source_language=language)
        assert en_result.text == english
        assert en_result.target_language == "en"
        assert en_result.source_language == language

        back = translator.to_vernacular(en_result.text, target_language=language)
        assert back.text == vern
        assert back.source_language == "en"
        assert back.target_language == language


# short-circuit paths
def test_english_input_is_passthrough(translator: OllamaTranslator, fake_llm: FakeLLM) -> None:
    result = translator.to_english("Hello world", source_language="en")
    assert result.text == "Hello world"
    assert fake_llm.calls == []


def test_empty_text_is_passthrough(translator: OllamaTranslator, fake_llm: FakeLLM) -> None:
    result = translator.to_vernacular("   ", target_language="hi")
    assert result.text == "   "
    assert fake_llm.calls == []


def test_english_target_is_passthrough(translator: OllamaTranslator, fake_llm: FakeLLM) -> None:
    result = translator.to_vernacular("Hello", target_language="en")
    assert result.text == "Hello"
    assert fake_llm.calls == []


def test_unsupported_source_raises(translator: OllamaTranslator) -> None:
    with pytest.raises(ValueError, match="Unsupported source language"):
        translator.to_english("bonjour", source_language="fr")


def test_unsupported_target_raises(translator: OllamaTranslator) -> None:
    with pytest.raises(ValueError, match="Unsupported target language"):
        translator.to_vernacular("hello", target_language="fr")


# prompt construction
def test_to_english_prompt_names_source_language(
    translator: OllamaTranslator, fake_llm: FakeLLM
) -> None:
    translator.to_english("नमस्ते", source_language="hi")
    assert fake_llm.calls
    system = fake_llm.calls[-1].messages[0]["content"]
    assert "Hindi" in system
    assert "English" in system


def test_quoted_output_is_stripped(fake_llm: FakeLLM) -> None:
    class QuotingLLM(FakeLLM):
        def complete(
            self,
            messages: list[dict[str, str]],
            model: str | None = None,
            json_mode: bool = False,
            think: bool = False,
            temperature: float | None = None,
            num_ctx: int | None = None,
            **_: object,  # absorb future kwargs
        ) -> LLMResponse:
            return LLMResponse(content='"Hello world"', model="x", usage={})

    translator = OllamaTranslator(llm=QuotingLLM({}))
    result = translator.to_english("नमस्ते दुनिया", source_language="hi")
    assert result.text == "Hello world"


# factory
def test_factory_default_returns_ollama() -> None:
    assert isinstance(get_translator(), OllamaTranslator)


# batch translation (corpus ingestion path)
class _NumberedLLM:
    """Stub that translates each [N] passage by prefixing 'EN:' and echoing it.

    Records every call so batching behaviour can be asserted.
    """

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        json_mode: bool = False,
        think: bool = False,
        temperature: float | None = None,
        num_ctx: int | None = None,
        **_: object,
    ) -> LLMResponse:
        self.calls.append(messages)
        prompt = messages[-1]["content"]
        out_lines: list[str] = []
        for line in prompt.splitlines():
            stripped = line.strip()
            match = re.match(r"^\[(\d+)\]\s*(.*)$", stripped)
            if match:
                out_lines.append(f"[{match.group(1)}] EN:{match.group(2)}")
        if not out_lines:
            # No [N] markers → this is the single-passage retry the ingest gate
            # fires when a batch translation came back still in Indic script.
            # Simulate a successful retry that returns real (ASCII) English.
            return LLMResponse(content="retried english", model=model or "fake", usage={})
        return LLMResponse(content="\n".join(out_lines), model=model or "fake", usage={})


def test_to_english_batch_aligns_and_translates() -> None:
    llm = _NumberedLLM()
    translator = OllamaTranslator(llm=llm)
    texts = ["Already English", "still English"]

    out = translator.to_english_batch(texts)

    assert out == ["EN:Already English", "EN:still English"]
    assert len(llm.calls) == 1  # all fit in one batch, none need a retry


def test_to_english_batch_retries_untranslated_passages() -> None:
    # The batch mock echoes Devanagari unchanged (a failed translation); the
    # ingest gate must detect it and retry each offending passage on its own so
    # vernacular text never lands in the English-only index.
    llm = _NumberedLLM()
    translator = OllamaTranslator(llm=llm)
    texts = ["नमस्ते", "Already English", "धन्यवाद"]

    out = translator.to_english_batch(texts)

    assert out == ["retried english", "EN:Already English", "retried english"]
    assert len(llm.calls) == 3  # 1 batch + 2 single-passage retries


def test_to_english_batch_splits_into_batches_of_ten() -> None:
    llm = _NumberedLLM()
    translator = OllamaTranslator(llm=llm)
    texts = [f"passage {i}" for i in range(23)]

    out = translator.to_english_batch(texts)

    assert out == [f"EN:passage {i}" for i in range(23)]
    assert len(llm.calls) == 3  # 10 + 10 + 3

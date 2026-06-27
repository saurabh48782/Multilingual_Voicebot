from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.graph.nodes.translate import make_to_english, make_to_vernacular
from src.translation.base import TranslationResult


def _en_node(translator: MagicMock | None = None):
    deps = MagicMock()
    if translator is not None:
        deps.translator = translator
    return make_to_english(deps)


def _vern_node(translator: MagicMock | None = None):
    deps = MagicMock()
    if translator is not None:
        deps.translator = translator
    return make_to_vernacular(deps)


def _tr(text: str, src: str, tgt: str) -> TranslationResult:
    return TranslationResult(text=text, source_language=src, target_language=tgt)


# make_to_english
@pytest.mark.parametrize(
    ("state", "translation", "expected_query"),
    [
        (
            {"transcript": "पीएम किसान योजना क्या है?", "source_language": "hi"},
            _tr("What is PM Kisan Yojana?", "hi", "en"),
            "What is PM Kisan Yojana?",
        ),
        (
            {"transcript": "পিএম কিষান প্রকল্প কী?", "source_language": "bn"},
            _tr("What is PM Kisan scheme?", "bn", "en"),
            "What is PM Kisan scheme?",
        ),
        (
            # source_language defaults to "en" when absent from state
            {"transcript": "How do I apply?"},
            _tr("How do I apply?", "en", "en"),
            "How do I apply?",
        ),
    ],
    ids=["hindi-to-english", "bengali-to-english", "default-lang-en"],
)
def test_to_english_translates(
    state: dict, translation: TranslationResult, expected_query: str
) -> None:
    translator = MagicMock()
    translator.to_english.return_value = translation
    result = _en_node(translator)(state)

    translator.to_english.assert_called_once_with(
        state.get("transcript", ""),
        source_language=state.get("source_language", "en"),
    )
    assert result == {"english_query": expected_query}


def test_to_english_empty_transcript_passes_through() -> None:
    translator = MagicMock()
    translator.to_english.return_value = _tr("", "hi", "en")
    result = _en_node(translator)({"transcript": "", "source_language": "hi"})
    assert result == {"english_query": ""}


@pytest.mark.parametrize(
    "exc",
    [RuntimeError("provider down"), OSError("timeout"), Exception("boom")],
)
def test_to_english_exception_falls_back_to_original(exc: Exception) -> None:
    translator = MagicMock()
    translator.to_english.side_effect = exc
    state = {"transcript": "यह क्या है?", "source_language": "hi"}
    result = _en_node(translator)(state)
    # falls back to the untranslated transcript — retrieval still has a chance
    assert result == {"english_query": "यह क्या है?"}


# make_to_vernacular
def test_to_vernacular_english_source_skips_translator() -> None:
    translator = MagicMock()
    state = {"english_response": "PM Kisan gives ₹6000/year.", "source_language": "en"}
    result = _vern_node(translator)(state)

    translator.to_vernacular.assert_not_called()
    assert result == {"vernacular_response": "PM Kisan gives ₹6000/year."}


@pytest.mark.parametrize(
    ("state", "translation", "expected"),
    [
        (
            {"english_response": "PM Kisan gives ₹6000/year.", "source_language": "hi"},
            _tr("पीएम किसान ₹6000 प्रति वर्ष देता है।", "en", "hi"),
            "पीएम किसान ₹6000 प्रति वर्ष देता है।",
        ),
        (
            {"english_response": "You must submit Aadhaar.", "source_language": "bn"},
            _tr("আপনাকে আধার জমা দিতে হবে।", "en", "bn"),
            "আপনাকে আধার জমা দিতে হবে।",
        ),
    ],
    ids=["english-to-hindi", "english-to-bengali"],
)
def test_to_vernacular_translates(
    state: dict, translation: TranslationResult, expected: str
) -> None:
    translator = MagicMock()
    translator.to_vernacular.return_value = translation
    result = _vern_node(translator)(state)

    translator.to_vernacular.assert_called_once_with(
        state["english_response"], target_language=state["source_language"]
    )
    assert result == {"vernacular_response": expected}


@pytest.mark.parametrize(
    "exc",
    [RuntimeError("provider down"), OSError("timeout"), Exception("boom")],
)
def test_to_vernacular_exception_falls_back_to_english(exc: Exception) -> None:
    translator = MagicMock()
    translator.to_vernacular.side_effect = exc
    state = {"english_response": "Apply at CSC centre.", "source_language": "hi"}
    result = _vern_node(translator)(state)
    # falls back to English answer — better than losing the response entirely
    assert result == {"vernacular_response": "Apply at CSC centre."}


def test_to_vernacular_missing_english_response_defaults_empty() -> None:
    translator = MagicMock()
    translator.to_vernacular.return_value = _tr("", "en", "hi")
    result = _vern_node(translator)({"source_language": "hi"})
    assert result == {"vernacular_response": ""}

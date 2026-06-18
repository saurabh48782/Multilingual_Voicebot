"""Unit tests for the numbered-translation response parser."""

from src.translation.base import _parse_numbered


def test_single_line_passages() -> None:
    raw = "[1] First translated.\n[2] Second translated."
    assert _parse_numbered(raw, 2, ["fb1", "fb2"]) == [
        "First translated.",
        "Second translated.",
    ]


def test_multiline_passage_is_kept_whole() -> None:
    raw = "[1] First line of passage one\nthat wraps onto a second line.\n---\n[2] Passage two."
    out = _parse_numbered(raw, 2, ["fb1", "fb2"])
    assert out[0] == "First line of passage one that wraps onto a second line."
    assert out[1] == "Passage two."


def test_missing_entry_uses_fallback() -> None:
    out = _parse_numbered("[1] only one came back", 2, ["fb1", "fb2"])
    assert out == ["only one came back", "fb2"]


def test_unnumbered_response_falls_back_entirely() -> None:
    out = _parse_numbered("model ignored the format completely", 2, ["a", "b"])
    assert out == ["a", "b"]


def test_empty_translation_uses_fallback() -> None:
    out = _parse_numbered("[1]\n[2] ok", 2, ["original one", "original two"])
    assert out == ["original one", "ok"]

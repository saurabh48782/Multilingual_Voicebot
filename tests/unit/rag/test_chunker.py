"""Unit tests for text chunking logic."""

from pathlib import Path
from typing import Any

import pytest

from src.rag import chunker as chunker_mod
from src.rag.chunker import (
    Chunk,
    _detect_lang,
    _is_dot_leader_table,
    _is_noise_chunk,
    _make_id,
    _paragraph_split,
    _split_sentences,
    chunk_pdf,
    chunk_text_file,
)

# --- Docling test doubles (no models, no downloads) ------------------------


class _FakeProv:
    def __init__(self, page_no: int) -> None:
        self.page_no = page_no


class _FakeItem:
    def __init__(self, label: str, page_no: int) -> None:
        self.label = label
        self.prov = [_FakeProv(page_no)]


class _FakeMeta:
    def __init__(self, headings: list[str], items: list[_FakeItem]) -> None:
        self.headings = headings
        self.doc_items = items


class _FakeDocChunk:
    def __init__(self, text: str, headings: list[str], label: str, page_no: int) -> None:
        self._text = text
        self.meta = _FakeMeta(headings, [_FakeItem(label, page_no)])


class _FakeHybridChunker:
    def __init__(self, chunks: list[_FakeDocChunk]) -> None:
        self._chunks = chunks

    def chunk(self, _doc: Any) -> list[_FakeDocChunk]:
        return self._chunks

    def contextualize(self, ch: _FakeDocChunk) -> str:
        return ch._text


def _patch_docling(monkeypatch: pytest.MonkeyPatch, chunks: list[_FakeDocChunk]) -> None:
    """Stub the two Docling seams so chunk_pdf runs offline."""
    monkeypatch.setattr(chunker_mod, "_convert_pdf", lambda _p: object())
    monkeypatch.setattr(chunker_mod, "_make_hybrid_chunker", lambda: _FakeHybridChunker(chunks))


def test_paragraph_split_empty() -> None:
    assert _paragraph_split("") == []


def test_paragraph_split_short_text() -> None:
    chunks = _paragraph_split("Hello world.")
    assert chunks == ["Hello world."]


def test_paragraph_split_respects_chunk_size() -> None:
    long_para = "word " * 300  # ~1500 chars
    chunks = _paragraph_split(long_para, chunk_size=500, overlap=0)
    assert all(len(c) <= 500 for c in chunks)


def test_paragraph_split_paragraphs_stay_together_if_small() -> None:
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = _paragraph_split(text, chunk_size=200, overlap=0)
    joined = " ".join(chunks)
    assert "Para one" in joined
    assert "Para two" in joined
    assert "Para three" in joined


def test_paragraph_split_overlap_prepends_tail() -> None:
    text = "A" * 400 + "\n\n" + "B" * 400
    chunks = _paragraph_split(text, chunk_size=450, overlap=50)
    assert len(chunks) == 2
    assert chunks[1].startswith(chunks[0][-50:].strip())


@pytest.mark.parametrize(
    ("args_a", "args_b", "equal"),
    [
        (("doc1", 0, "some text"), ("doc1", 0, "some text"), True),
        (("doc1", 0, "text A"), ("doc1", 0, "text B"), False),
    ],
    ids=["deterministic", "differs_by_text"],
)
def test_make_id(args_a: tuple[str, int, str], args_b: tuple[str, int, str], equal: bool) -> None:
    assert (_make_id(*args_a) == _make_id(*args_b)) is equal


def test_chunk_id_set_on_init() -> None:
    c = Chunk(
        doc_id="d",
        chunk_index=0,
        text="hello",
        text_en="hello",
        source="f.txt",
        page_num=-1,
    )
    assert len(c.chunk_id) == 16
    assert isinstance(c.chunk_id_int, int)
    assert 0 <= c.chunk_id_int < 2**63


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("प्रधानमंत्री किसान योजना", "hi"),
        ("প্রধানমন্ত্রী", "bn"),
        ("English text only", "en"),
    ],
    ids=["devanagari", "bengali", "english"],
)
def test_detect_lang(text: str, expected: str) -> None:
    assert _detect_lang(text) == expected


def test_chunk_text_file_doc_id_is_full_filename(tmp_path: Path) -> None:
    """doc_id must include the extension so `a.pdf` and `a.txt` never collide
    in the index (previously `path.stem` made them the same doc_id)."""
    path = tmp_path / "a.txt"
    path.write_text("Hello world.")
    chunks = chunk_text_file(path)
    assert chunks[0].doc_id == "a.txt"


def test_chunk_pdf_doc_id_is_full_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_docling(monkeypatch, [_FakeDocChunk("Hello world.", ["Intro"], "text", 1)])
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4")  # never parsed - _convert_pdf is stubbed

    chunks = chunk_pdf(path)
    assert chunks[0].doc_id == "a.pdf"


def test_chunk_pdf_captures_headings_and_content_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_docling(
        monkeypatch,
        [
            _FakeDocChunk(
                "Some prose about the scheme.", ["2. Benefits", "2.1 Eligibility"], "text", 3
            ),
            _FakeDocChunk("| YEAR | SCHEME |\n| - | - |\n| 1952 | CDP |", [], "table", 4),
        ],
    )
    chunks = chunk_pdf(tmp_path / "a.pdf")

    assert len(chunks) == 2
    prose, table = chunks
    assert prose.content_type == "text"
    assert prose.headings == "2. Benefits > 2.1 Eligibility"
    assert prose.page_num == 3
    assert table.content_type == "table"
    assert "| YEAR | SCHEME |" in table.text
    assert table.page_num == 4
    # chunk_index is reassigned densely after filtering
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_chunk_pdf_drops_noise_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toc = f"| 1 INTRODUCTION {'.' * 40} 5 | 2 SCOPE {'.' * 40} 8 |\n| - | - |"
    _patch_docling(
        monkeypatch,
        [
            _FakeDocChunk("<!-- image -->", [], "picture", 1),  # picture-only
            _FakeDocChunk("   ", [], "text", 1),  # empty
            _FakeDocChunk(toc, [], "table", 1),  # TOC dot-leader table
            _FakeDocChunk("Real content.", ["Body"], "text", 2),  # kept
        ],
    )
    chunks = chunk_pdf(tmp_path / "a.pdf")

    assert len(chunks) == 1
    assert chunks[0].text == "Real content."


def test_noise_and_dot_leader_helpers() -> None:
    assert _is_noise_chunk("<!-- image -->", "text") is True
    assert _is_noise_chunk("   ", "table") is True
    assert _is_noise_chunk("Real content.", "text") is False
    assert _is_dot_leader_table("| 1 INTRO ......... 5 |\n| - |") is True
    assert _is_dot_leader_table("| YEAR | SCHEME |\n| 1952 | CDP |") is False


def test_split_sentences_english() -> None:
    out = _split_sentences("First sentence. Second one! Third sentence.")
    assert len(out) >= 2
    assert out[0].startswith("First")

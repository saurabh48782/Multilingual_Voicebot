"""Unit tests for text chunking logic."""

from pathlib import Path

import fitz
import pytest

from src.rag.chunker import (
    Chunk,
    _detect_lang,
    _make_id,
    _paragraph_split,
    _split_sentences,
    chunk_pdf,
    chunk_text_file,
)


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


def test_chunk_pdf_doc_id_is_full_filename(tmp_path: Path) -> None:
    path = tmp_path / "a.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello world.")
    doc.save(str(path))
    doc.close()

    chunks = chunk_pdf(path)
    assert chunks[0].doc_id == "a.pdf"


def test_split_sentences_english() -> None:
    out = _split_sentences("First sentence. Second one! Third sentence.")
    assert len(out) >= 2
    assert out[0].startswith("First")

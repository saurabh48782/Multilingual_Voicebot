"""Split documents into semantically coherent chunks.

Strategies (selected via `cfg["rag"]["chunking"]["strategy"]`):
  - "semantic": embed each sentence with the same e5 model used downstream,
                split at large adjacent-sentence cosine-distance jumps
                (percentile-based), then enforce min/max chunk size.
  - "paragraph": legacy paragraph + sentence splitter (kept for fallback).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

from src.utils.config import cfg

# Paragraph-mode defaults (legacy)
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str  # original language
    text_en: str  # filled by ingestor after translation
    source: str  # file path str
    page_num: int  # 0-indexed, -1 for plain text
    chunk_id: str = field(init=False)
    chunk_id_int: int = field(init=False)

    def __post_init__(self) -> None:
        self.chunk_id, self.chunk_id_int = _make_id(
            self.doc_id, self.chunk_index, self.text_en or self.text
        )


def _make_id(doc_id: str, chunk_index: int, text: str) -> tuple[str, int]:
    hex16 = hashlib.sha256(f"{doc_id}:{chunk_index}:{text}".encode()).hexdigest()[:16]
    return hex16, int(hex16, 16) % (2**63)


# ---------------------------------------------------------------------------
# Language / sentence segmentation
# ---------------------------------------------------------------------------

_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("hi", 0x0900, 0x097F),  # Devanagari (Hindi)
    ("bn", 0x0980, 0x09FF),  # Bengali
)


def _detect_lang(text: str) -> str:
    """Cheap script-based language hint. Falls back to 'en'."""
    for ch in text[:1000]:
        cp = ord(ch)
        for code, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                return code
    return "en"


_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?।॥])\s+")


def _split_sentences(text: str) -> list[str]:
    """Sentence split via indic-nlp-library when available, regex fallback."""
    text = text.strip()
    if not text:
        return []
    lang = _detect_lang(text)
    try:
        from indicnlp.tokenize import (  # type: ignore[import-untyped, unused-ignore]
            sentence_tokenize,
        )

        sents = sentence_tokenize.sentence_split(text, lang=lang)
    except Exception:
        sents = _SENT_BOUNDARY_RE.split(text)
    return [s.strip() for s in sents if s and s.strip()]


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------


def _semantic_split(text: str) -> list[str]:
    """Sentence-level semantic chunking.

    1. Split into sentences.
    2. Embed each sentence.
    3. Compute cosine *distance* between adjacent sentences.
    4. Pick breakpoints where distance exceeds the configured percentile.
    5. Merge adjacent sentences inside each segment; enforce min/max char bounds.
    """
    cfg_chunk = cfg["rag"]["chunking"]
    min_size = int(cfg_chunk["min_chunk_size"])
    max_size = int(cfg_chunk["max_chunk_size"])
    pct = float(cfg_chunk["breakpoint_percentile"])
    overlap_sents = int(cfg_chunk.get("overlap_sentences", 0))

    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return _enforce_max_size(sentences, max_size)

    # Local import to avoid heavy load when only paragraph mode is used
    from src.rag.embedder import embed_passages

    vecs = embed_passages(sentences, batch_size=32)
    sims = (vecs[:-1] * vecs[1:]).sum(axis=1)
    distances = 1.0 - sims  # (N-1,) cosine distances between consecutive sentences

    if distances.size == 0:
        threshold = float("inf")
    else:
        threshold = float(np.percentile(distances, pct))

    # Build raw segments delimited by breakpoint indices
    segments: list[list[str]] = []
    current: list[str] = [sentences[0]]
    for i, dist in enumerate(distances, start=1):
        if dist >= threshold and sum(len(s) for s in current) >= min_size:
            segments.append(current)
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        segments.append(current)

    # Merge tiny tail segments forward to satisfy min_size
    merged: list[list[str]] = []
    for seg in segments:
        if merged and sum(len(s) for s in merged[-1]) < min_size:
            merged[-1].extend(seg)
        else:
            merged.append(seg)

    chunks = [" ".join(seg).strip() for seg in merged]
    chunks = _enforce_max_size(chunks, max_size)

    if overlap_sents > 0 and len(chunks) > 1:
        chunks = _apply_sentence_overlap(chunks, overlap_sents)
    return chunks


def _enforce_max_size(chunks: list[str], max_size: int) -> list[str]:
    """Sub-split any chunk longer than max_size on sentence boundaries."""
    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_size:
            out.append(chunk)
            continue
        sents = _split_sentences(chunk)
        buf: list[str] = []
        size = 0
        for s in sents:
            if size + len(s) + 1 > max_size and buf:
                out.append(" ".join(buf).strip())
                buf, size = [], 0
            buf.append(s)
            size += len(s) + 1
        if buf:
            out.append(" ".join(buf).strip())
    return out


def _apply_sentence_overlap(chunks: list[str], n: int) -> list[str]:
    """Prepend the last `n` sentences of the previous chunk to the next."""
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_sents = _split_sentences(chunks[i - 1])
        tail = " ".join(prev_sents[-n:]) if prev_sents else ""
        overlapped.append((tail + " " + chunks[i]).strip() if tail else chunks[i])
    return overlapped


# ---------------------------------------------------------------------------
# Paragraph chunking (legacy fallback)
# ---------------------------------------------------------------------------


def _paragraph_split(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split on paragraph/sentence boundaries, respecting chunk_size."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                buf = ""
                for sent in sentences:
                    if len(buf) + len(sent) + 1 <= chunk_size:
                        buf = (buf + " " + sent).strip() if buf else sent
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = sent[-chunk_size:] if len(sent) > chunk_size else sent
                if buf:
                    current = buf
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped

    return chunks


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------


def _split_text(text: str) -> list[str]:
    strategy = cfg["rag"].get("chunking", {}).get("strategy", "semantic")
    if strategy == "paragraph":
        return _paragraph_split(text)
    return _semantic_split(text)


# ---------------------------------------------------------------------------
# File-level entrypoints (unchanged signatures)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _norm_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _page_starts(page_texts: list[str]) -> list[int]:
    """Cumulative start offsets of each page in the whitespace-normalised
    concatenation of all pages (single-space-joined)."""
    starts: list[int] = []
    offset = 0
    for text in page_texts:
        starts.append(offset)
        norm = _norm_ws(text)
        offset += len(norm) + 1  # +1 for the join space
    return starts


def chunk_pdf(path: Path, doc_id: str | None = None) -> list[Chunk]:
    """Chunk the whole document at once so passages spanning page breaks stay
    together; each chunk is attributed to the page where it starts.
    `sort=True` orders blocks by position, which fixes reading order on the
    multi-column layouts common in government PDFs."""
    doc_id = doc_id or path.stem

    with fitz.open(str(path)) as doc:
        page_texts = [page.get_text(sort=True) for page in doc]

    full_text = "\n\n".join(page_texts)
    pieces = _split_text(full_text)
    if not pieces:
        return []

    norm_full = " ".join(_norm_ws(t) for t in page_texts)
    starts = _page_starts(page_texts)

    chunks: list[Chunk] = []
    search_from = 0
    for chunk_index, piece in enumerate(pieces):
        # Locate the chunk by a normalised prefix; monotonic search keeps
        # overlap-duplicated sentences anchored to their true position.
        probe = _norm_ws(piece)[:60]
        pos = norm_full.find(probe, search_from) if probe else -1
        if pos == -1:
            pos = norm_full.find(probe) if probe else -1
        if pos >= 0:
            search_from = pos
            page_num = max(i for i, s in enumerate(starts) if s <= pos)
        else:
            page_num = chunks[-1].page_num if chunks else 0
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_index=chunk_index,
                text=piece,
                text_en="",
                source=str(path),
                page_num=page_num,
            )
        )
    return chunks


def chunk_text_file(path: Path, doc_id: str | None = None) -> list[Chunk]:
    doc_id = doc_id or path.stem
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        Chunk(
            doc_id=doc_id,
            chunk_index=i,
            text=piece,
            text_en="",
            source=str(path),
            page_num=-1,
        )
        for i, piece in enumerate(_split_text(text))
    ]


def chunk_file(path: Path, doc_id: str | None = None) -> list[Chunk]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return chunk_pdf(path, doc_id)
    if suffix == ".txt":
        return chunk_text_file(path, doc_id)
    raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt")

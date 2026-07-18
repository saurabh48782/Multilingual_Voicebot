"""Split documents into retrievable chunks.

PDFs are parsed layout-aware with Docling (structure, tables, reading order)
and chunked with Docling's structure-aware HybridChunker: section headings are
captured as a metadata breadcrumb, tables are serialised to Markdown and kept
intact, and chunk sizes respect the embedder's token budget.

Plain-text (.txt) files use the sentence-level chunker selected via
`cfg["rag"]["chunking"]["strategy"]`:
  - "semantic": embed each sentence with the same e5 model used downstream,
                split at large adjacent-sentence cosine-distance jumps
                (percentile-based), then enforce min/max chunk size.
  - "paragraph": legacy paragraph + sentence splitter (kept for fallback).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

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
    page_num: int  # 1-indexed page (Docling PDFs), -1 for plain text
    headings: str = ""  # " > "-joined section-heading breadcrumb (Docling)
    content_type: str = "text"  # "text" | "table"
    chunk_id: str = field(init=False)
    chunk_id_int: int = field(init=False)

    def __post_init__(self) -> None:
        self.chunk_id, self.chunk_id_int = _make_id(
            self.doc_id, self.chunk_index, self.text_en or self.text
        )


def _make_id(doc_id: str, chunk_index: int, text: str) -> tuple[str, int]:
    hex16 = hashlib.sha256(f"{doc_id}:{chunk_index}:{text}".encode()).hexdigest()[:16]
    return hex16, int(hex16, 16) % (2**63)


# Language / sentence segmentation
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


# Semantic chunking
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
    """Sub-split any chunk longer than max_size on sentence boundaries.

    A single sentence that itself exceeds max_size (no punctuation to split
    on) is hard-split on word boundaries instead of being emitted whole,
    which would otherwise be silently truncated by the embedder's token limit.
    """
    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_size:
            out.append(chunk)
            continue
        sents = _split_sentences(chunk)
        buf: list[str] = []
        size = 0
        for s in sents:
            if len(s) > max_size:
                if buf:
                    out.append(" ".join(buf).strip())
                    buf, size = [], 0
                out.extend(_hard_split(s, max_size))
                continue
            if size + len(s) + 1 > max_size and buf:
                out.append(" ".join(buf).strip())
                buf, size = [], 0
            buf.append(s)
            size += len(s) + 1
        if buf:
            out.append(" ".join(buf).strip())
    return out


def _hard_split(text: str, max_size: int) -> list[str]:
    """Split a boundary-less sentence on word boundaries so no piece exceeds
    max_size; a single word longer than max_size is cut on characters."""
    out: list[str] = []
    buf = ""
    for word in text.split(" "):
        while len(word) > max_size:
            if buf:
                out.append(buf)
                buf = ""
            out.append(word[:max_size])
            word = word[max_size:]
        if buf and len(buf) + len(word) + 1 > max_size:
            out.append(buf)
            buf = word
        else:
            buf = f"{buf} {word}".strip()
    if buf:
        out.append(buf)
    return out


def _apply_sentence_overlap(chunks: list[str], n: int) -> list[str]:
    """Prepend the last `n` sentences of the previous chunk to the next."""
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_sents = _split_sentences(chunks[i - 1])
        tail = " ".join(prev_sents[-n:]) if prev_sents else ""
        overlapped.append((tail + " " + chunks[i]).strip() if tail else chunks[i])
    return overlapped


# Paragraph chunking (legacy fallback)
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


# Strategy dispatch
def _split_text(text: str) -> list[str]:
    strategy = cfg["rag"].get("chunking", {}).get("strategy", "semantic")
    if strategy == "paragraph":
        return _paragraph_split(text)
    return _semantic_split(text)


# Docling: layout-aware PDF parse + structure-aware chunking
_IMAGE_PLACEHOLDER_RE = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)


def _docling_pipeline_options() -> Any:
    """Build `PdfPipelineOptions` from `cfg["rag"]["chunking"]["docling"]`."""
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

    dcfg = cfg["rag"].get("chunking", {}).get("docling", {})
    opts = PdfPipelineOptions()
    opts.do_table_structure = True
    table_options: Any = opts.table_structure_options
    table_options.do_cell_matching = True
    mode = str(dcfg.get("table_mode", "accurate")).lower()
    table_options.mode = TableFormerMode.FAST if mode == "fast" else TableFormerMode.ACCURATE
    opts.do_ocr = bool(dcfg.get("do_ocr", False))
    return opts


def _convert_pdf(path: Path) -> Any:
    """Parse a PDF into a `DoclingDocument` (layout, tables, reading order).

    Isolated as a module-level seam so unit tests can monkeypatch it and skip
    the heavy model download / conversion.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_docling_pipeline_options())
        }
    )
    return converter.convert(str(path)).document


def _markdown_serializer_provider() -> Any:
    """Serializer that emits tables as Markdown pipe-tables inside chunks
    (HybridChunker otherwise flattens them to text). Returns None if the
    Docling serializer API is unavailable."""
    try:
        from docling_core.transforms.chunker.hierarchical_chunker import (
            ChunkingDocSerializer,
            ChunkingSerializerProvider,
        )
        from docling_core.transforms.serializer.markdown import (
            MarkdownParams,
            MarkdownTableSerializer,
        )
    except Exception:  # noqa: BLE001 - optional API, degrade to flat tables
        logger.warning("Docling Markdown table serializer unavailable; tables may be flattened")
        return None

    def get_serializer(self: object, doc: Any) -> Any:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True),
        )

    markdown_tables_cls = type(
        "_MarkdownTables",
        (ChunkingSerializerProvider,),
        {"get_serializer": get_serializer},
    )
    return markdown_tables_cls()


def _make_hybrid_chunker() -> Any:
    """HybridChunker sized to the e5 token budget, emitting Markdown tables."""
    docling_chunking: Any = import_module("docling.chunking")
    hybrid_chunker_cls = docling_chunking.HybridChunker

    dcfg = cfg["rag"].get("chunking", {}).get("docling", {})
    kwargs: dict[str, Any] = {
        "max_tokens": int(dcfg.get("max_tokens", 400)),
        "merge_peers": True,
        "repeat_table_header": True,
    }
    # Size chunks against the same tokenizer the embedder uses (512-token e5),
    # so a chunk never silently overflows and gets truncated at embed time.
    try:
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

        kwargs["tokenizer"] = HuggingFaceTokenizer.from_pretrained(cfg["rag"]["embedding_model"])
    except Exception:  # noqa: BLE001 - fall back to HybridChunker's default tokenizer
        logger.warning("Docling e5 tokenizer unavailable; using HybridChunker default")
    provider = _markdown_serializer_provider()
    if provider is not None:
        kwargs["serializer_provider"] = provider
    return hybrid_chunker_cls(**kwargs)


def _chunk_meta(dc: Any, fallback_page: int) -> tuple[str, int]:
    """Return (content_type, page_num) for a Docling chunk.

    content_type is "table" when any source doc item is a table; page_num is
    the first (lowest) page the chunk's items appear on (Docling pages are
    1-indexed), falling back to the previous chunk's page when provenance is
    absent.
    """
    items = getattr(dc.meta, "doc_items", None) or []
    is_table = False
    page: int | None = None
    for it in items:
        if "table" in str(getattr(it, "label", "")).lower():
            is_table = True
        for prov in getattr(it, "prov", None) or []:
            p = getattr(prov, "page_no", None)
            if p is not None and (page is None or int(p) < page):
                page = int(p)
    return ("table" if is_table else "text"), (page if page is not None else fallback_page)


def _is_dot_leader_table(md: str) -> bool:
    """True for a table-of-contents rendered as a table (cells are mostly dot
    leaders) - TableFormer over-detects these on dotted TOC lines."""
    cell_text = md.replace("|", " ").replace("-", " ")
    non_space = [c for c in cell_text if not c.isspace()]
    if not non_space:
        return True
    dots = sum(1 for c in non_space if c == ".")
    return dots / len(non_space) > 0.5


def _is_noise_chunk(text: str, content_type: str) -> bool:
    """Drop empty / picture-only chunks and TOC dot-leader tables."""
    stripped = _IMAGE_PLACEHOLDER_RE.sub("", text).strip()
    if not stripped:
        return True
    return content_type == "table" and _is_dot_leader_table(text)


def chunk_pdf(path: Path, doc_id: str | None = None) -> list[Chunk]:
    """Layout-aware PDF chunking via Docling + HybridChunker.

    Tables are preserved as Markdown, section headings are captured as a
    breadcrumb in chunk metadata, and chunk sizes respect the e5 token budget.
    Empty/picture-only chunks and table-of-contents dot-leader tables are
    dropped.
    """
    doc_id = doc_id or path.name
    document = _convert_pdf(path)
    chunker = _make_hybrid_chunker()

    chunks: list[Chunk] = []
    last_page = 1
    index = 0
    for dc in chunker.chunk(document):
        text = chunker.contextualize(dc).strip()
        content_type, page_num = _chunk_meta(dc, last_page)
        last_page = page_num
        if _is_noise_chunk(text, content_type):
            continue
        headings = " > ".join(getattr(dc.meta, "headings", None) or [])
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_index=index,
                text=text,
                text_en="",
                source=str(path),
                page_num=page_num,
                headings=headings,
                content_type=content_type,
            )
        )
        index += 1
    return chunks


def chunk_text_file(path: Path, doc_id: str | None = None) -> list[Chunk]:
    doc_id = doc_id or path.name
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

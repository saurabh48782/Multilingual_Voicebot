"""Corpus ingestion pipeline.

Flow per file:
  1. Compute SHA-256 file hash; skip if manifest shows it's unchanged.
  2. Chunk into passages (PDF page-aware or plain-text).
  3. Translate non-English passages to English via the configured translation
     provider (Ollama via translategemma).
  4. Embed with multilingual-e5-large (passage prefix, L2-normalised).
  5. Upsert into FAISS store with metadata.
  6. Persist index + parquet + manifest.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path

from src.rag import bm25_store as _bm25_mod
from src.rag import store as _store_mod
from src.rag.bm25_store import BM25Store, get_bm25_store
from src.rag.chunker import chunk_file
from src.rag.embedder import embed_passages
from src.rag.locks import index_rwlock
from src.rag.store import FAISSStore, get_store
from src.utils.config import (
    bm25_corpus_path,
    bm25_index_dir,
    faiss_index_path,
    faiss_manifest_path,
    faiss_metadata_path,
)
from src.utils.config import corpus_dir as _default_corpus_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

# Serialises whole ingest runs (manifest read-modify-write is not atomic);
# index_rwlock.write() additionally excludes concurrent searches during the
# actual index mutation. Process-local - see src/rag/locks.py.
_INGEST_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict[str, str]:
    if faiss_manifest_path.exists():
        manifest: dict[str, str] = json.loads(faiss_manifest_path.read_text())
        return manifest
    return {}


def _save_manifest(manifest: dict[str, str]) -> None:
    faiss_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_manifest_path.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Index teardown
# ---------------------------------------------------------------------------


def clear_index() -> list[Path]:
    """Delete every on-disk FAISS + BM25 artifact and the ingestion manifest.
    Returns the artifact paths that existed and were removed.
    """
    artifacts = [
        faiss_index_path,
        faiss_metadata_path,
        faiss_manifest_path,
        bm25_corpus_path,
        bm25_index_dir,
    ]
    removed: list[Path] = []
    with _INGEST_LOCK, index_rwlock.write():
        for path in artifacts:
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path)
        # Drop cached singletons; the next get_store()/get_bm25_store() rebuilds
        # an empty index from the now-missing files.
        _store_mod._store = None
        _bm25_mod._store = None
    logger.info("Cleared index", artifacts_removed=len(removed))
    return removed


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


def _ingest_file(
    path: Path,
    store: FAISSStore,
    bm25: BM25Store,
    force: bool = False,
    translate: bool = True,
) -> tuple[int, str]:
    """Ingest one file. Returns (chunks_added, status).

    status ∈ {'added', 'updated', 'skipped'}.
    """
    manifest = _load_manifest()
    file_key = str(path.resolve())
    current_hash = _file_hash(path)

    if not force and manifest.get(file_key) == current_hash:
        logger.info("Skipping %s (unchanged)", path.name)
        return 0, "skipped"

    # Stale-chunk removal is deferred into the write-locked window below so
    # searches never observe a half-removed document.
    needs_removal = file_key in manifest
    status = "updated" if needs_removal else "added"

    chunks = chunk_file(path)
    if not chunks:
        logger.warning("No chunks extracted from %s", path.name)
        if needs_removal:
            with index_rwlock.write():
                removed = store.remove_doc(path.stem)
                bm25.remove_doc(path.stem)
            logger.info("Removed %d stale chunks for %s", removed, path.name)
        manifest[file_key] = current_hash
        _save_manifest(manifest)
        return 0, status

    logger.info("Chunked %s → %d passages", path.name, len(chunks))

    # Translate
    if translate:
        from src.translation import get_translator

        original_texts = [c.text for c in chunks]
        translated = get_translator().to_english_batch(original_texts)
        for chunk, en in zip(chunks, translated):
            chunk.text_en = en
            # Recompute IDs now that text_en is set
            from src.rag.chunker import _make_id

            chunk.chunk_id, chunk.chunk_id_int = _make_id(
                chunk.doc_id, chunk.chunk_index, chunk.text_en
            )
    else:
        for chunk in chunks:
            chunk.text_en = chunk.text
            from src.rag.chunker import _make_id

            chunk.chunk_id, chunk.chunk_id_int = _make_id(
                chunk.doc_id, chunk.chunk_index, chunk.text_en
            )

    # Embed
    logger.info("Embedding %d passages…", len(chunks))
    vecs = embed_passages([c.text_en for c in chunks])

    # Build metadata rows
    meta_rows = [
        {
            "chunk_id": c.chunk_id,
            "chunk_id_int": c.chunk_id_int,
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "text_en": c.text_en,
            "source": c.source,
            "page_num": c.page_num,
        }
        for c in chunks
    ]

    with index_rwlock.write():
        if needs_removal:
            removed = store.remove_doc(path.stem)
            bm25.remove_doc(path.stem)
            logger.info("Removed %d stale chunks for %s", removed, path.name)

        store.upsert(
            chunk_ids_int=[c.chunk_id_int for c in chunks],
            chunk_ids_hex=[c.chunk_id for c in chunks],
            vectors=vecs,
            meta_rows=meta_rows,
        )
        bm25.upsert([(c.chunk_id_int, c.doc_id, c.text_en) for c in chunks])

    manifest[file_key] = current_hash
    _save_manifest(manifest)
    logger.info("Upserted %d chunks for %s (FAISS + BM25)", len(chunks), path.name)
    return len(chunks), status


def ingest_corpus(
    corpus_dir: Path | None = None,
    force: bool = False,
    translate: bool = True,
) -> dict[str, int]:
    """Ingest all supported files in corpus_dir.

    Returns {filename: chunks_added}.
    """
    corpus_dir = corpus_dir or _default_corpus_dir
    store = get_store()
    bm25 = get_bm25_store()

    files = [
        p
        for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        logger.warning("No supported files found in %s", corpus_dir)
        return {}

    summary: dict[str, int] = {}
    with _INGEST_LOCK:
        for path in sorted(files):
            try:
                count, status = _ingest_file(
                    path, store, bm25, force=force, translate=translate
                )
                summary[path.name] = count
                logger.info("[%s] %s - %d chunks", status.upper(), path.name, count)
            except Exception:
                logger.exception("Failed to ingest %s", path.name)

        with index_rwlock.write():
            store.save()
            bm25.save()
    logger.info(
        "Ingestion complete. Index size: %d FAISS / %d BM25 chunks.",
        store.total_chunks,
        bm25.total_chunks,
    )
    return summary


def ingest_file(
    path: Path,
    force: bool = False,
    translate: bool = True,
) -> int:
    """Ingest a single file. Returns chunks added."""
    store = get_store()
    bm25 = get_bm25_store()
    with _INGEST_LOCK:
        count, status = _ingest_file(
            path, store, bm25, force=force, translate=translate
        )
        with index_rwlock.write():
            store.save()
            bm25.save()
    logger.info("[%s] %s - %d chunks", status.upper(), path.name, count)
    return count

"""POST /api/documents - upload a PDF/TXT file and index it for RAG."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.schemas import (
    DocumentUploadResponse,
    IndexResetResponse,
    IndexStats,
)
from src.rag.ingestor import (
    SUPPORTED_EXTENSIONS,
    index_stats,
    ingest_file,
    reset_index,
)
from src.utils.config import DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["Documents"])

_UPLOAD_DIR = DATA_DIR / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


def safe_destination(filename: str | None, upload_dir: Path = _UPLOAD_DIR) -> Path:
    """Resolve an upload filename to a path strictly inside upload_dir.

    Re-uploading the same name intentionally overwrites: doc_id is derived from
    the full filename, so ingestion replaces the previous chunks instead of
    duplicating them.
    """
    name = Path(filename or "").name
    if not name or name.startswith(".") or "\\" in name or name != name.strip():
        raise HTTPException(status_code=422, detail="Invalid filename")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    dest = (upload_dir / name).resolve()
    if dest.parent != upload_dir.resolve():
        raise HTTPException(status_code=422, detail="Invalid filename")
    return dest


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    summary="Upload documents",
    description="Upload a document to be indexed in the Vector Store",
)
async def upload_document(
    file: Annotated[UploadFile, File()],
) -> DocumentUploadResponse:
    dest = safe_destination(file.filename)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload")
    if len(raw) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")

    # Write off the event loop - a 50 MB blocking write would stall every other
    # in-flight request on the single-worker server.
    await asyncio.get_event_loop().run_in_executor(None, dest.write_bytes, raw)
    logger.info("Document uploaded", filename=dest.name, bytes=len(raw))

    try:
        chunks_added = await asyncio.get_event_loop().run_in_executor(
            None, lambda: ingest_file(dest, force=True, translate=True)
        )
    except Exception as exc:
        logger.exception("Ingestion failed", filename=dest.name)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return DocumentUploadResponse(
        filename=dest.name,
        chunks_added=chunks_added,
        message=f"Indexed {chunks_added} chunks from '{dest.name}'.",
    )


@router.get(
    "/documents/stats",
    response_model=IndexStats,
    summary="Vector Store stats",
    description="Fetch stats for all documents indexed in the Vector Store",
)
async def documents_stats() -> IndexStats:
    """Per-document chunk counts for the RAG index (mirrors the CLI --stats)."""
    stats = await asyncio.get_event_loop().run_in_executor(None, index_stats)
    return IndexStats.model_validate(stats)


@router.delete(
    "/documents",
    response_model=IndexResetResponse,
    summary="Clear Vector Store",
    description="Delete all documents indexed in the Vector Store",
)
async def reset_documents() -> IndexResetResponse:
    """Wipe the entire RAG index (FAISS + BM25 + manifest) and reload the live
    stores empty (mirrors the CLI --clear, but safe on a running server)."""
    removed = await asyncio.get_event_loop().run_in_executor(None, reset_index)
    logger.info("Index reset via API", artifacts_removed=len(removed))
    return IndexResetResponse(
        artifacts_removed=len(removed),
        total_chunks=0,
        message=f"Index wiped; removed {len(removed)} artifact(s).",
    )

"""RAG subsystem: chunking, embedding, FAISS store, ingestion."""

from src.rag.embedder import embed_passages, embed_query
from src.rag.ingestor import ingest_corpus, ingest_file
from src.rag.store import SearchResult, get_store

__all__ = [
    "embed_passages",
    "embed_query",
    "ingest_corpus",
    "ingest_file",
    "get_store",
    "SearchResult",
]

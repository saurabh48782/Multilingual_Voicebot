"""FAISS index manager.

Index type: IndexIDMap2(IndexFlatIP)
  - IndexFlatIP: exact inner product (= cosine when vecs are L2-normalised)
  - IndexIDMap2: maps arbitrary int64 IDs to internal FAISS rows; supports remove_ids

Metadata lives in a parquet file alongside the index file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.rag.embedder import EMBEDDING_DIM
from src.utils.config import cfg, faiss_index_path, faiss_metadata_path


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text_en: str
    source: str
    page_num: int
    score: float


class FAISSStore:
    def __init__(
        self,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        self.index_path = index_path or faiss_index_path
        self.metadata_path = metadata_path or faiss_metadata_path
        self._index: faiss.IndexIDMap2 | None = None
        self._meta: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Index access
    # ------------------------------------------------------------------

    @property
    def index(self) -> faiss.IndexIDMap2:
        if self._index is None:
            self._index = self._load_or_create()
        return self._index

    def _load_or_create(self) -> faiss.IndexIDMap2:
        if self.index_path.exists():
            idx = faiss.read_index(str(self.index_path))
        else:
            flat = faiss.IndexFlatIP(EMBEDDING_DIM)
            idx = faiss.IndexIDMap2(flat)
        return idx

    # ------------------------------------------------------------------
    # Metadata access
    # ------------------------------------------------------------------

    def _load_meta(self) -> None:
        if self.metadata_path.exists():
            self._meta = pd.read_parquet(self.metadata_path)
        else:
            self._meta = _empty_meta()

    def _save_meta(self) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta.to_parquet(self.metadata_path, index=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        self._index = self._load_or_create()
        self._load_meta()

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self._save_meta()

    def upsert(
        self,
        chunk_ids_int: list[int],
        chunk_ids_hex: list[str],
        vectors: np.ndarray,
        meta_rows: list[dict],
    ) -> None:
        """Add or replace vectors by ID. Existing IDs are removed first."""
        if not chunk_ids_int:
            return

        existing_ids = (
            set(self._meta["chunk_id"].tolist()) if len(self._meta) else set()
        )
        ids_to_remove = [
            cid_int
            for cid_int, cid_hex in zip(chunk_ids_int, chunk_ids_hex)
            if cid_hex in existing_ids
        ]
        if ids_to_remove:
            self.index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
            self._meta = self._meta[
                ~self._meta["chunk_id"].isin(
                    [cid_hex for cid_hex in chunk_ids_hex if cid_hex in existing_ids]
                )
            ]

        ids_np = np.array(chunk_ids_int, dtype=np.int64)
        self.index.add_with_ids(vectors, ids_np)

        new_rows = pd.DataFrame(meta_rows)
        self._meta = pd.concat([self._meta, new_rows], ignore_index=True)

    def remove_doc(self, doc_id: str) -> int:
        """Remove all chunks for doc_id. Returns count removed."""
        rows = self._meta[self._meta["doc_id"] == doc_id]
        if rows.empty:
            return 0
        ids_int = rows["chunk_id_int"].tolist()
        self.index.remove_ids(np.array(ids_int, dtype=np.int64))
        self._meta = self._meta[self._meta["doc_id"] != doc_id]
        return len(ids_int)

    def search(
        self, query_vec: np.ndarray, top_k: int | None = None
    ) -> list[SearchResult]:
        """Return top-k results sorted by cosine score descending."""
        k = top_k or cfg["rag"]["top_k"]
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(query_vec, k)
        scores = scores[0]
        ids = ids[0]

        id_to_meta = {row["chunk_id_int"]: row for _, row in self._meta.iterrows()}
        results: list[SearchResult] = []
        for score, fid in zip(scores, ids):
            if fid == -1:
                continue
            row = id_to_meta.get(int(fid))
            if row is None:
                continue
            results.append(
                SearchResult(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    chunk_index=int(row["chunk_index"]),
                    text_en=row["text_en"],
                    source=row["source"],
                    page_num=int(row["page_num"]),
                    score=float(score),
                )
            )
        return results

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal


def _empty_meta() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "chunk_id",
            "chunk_id_int",
            "doc_id",
            "chunk_index",
            "text_en",
            "source",
            "page_num",
        ]
    )


# Module-level singleton - loaded lazily
_store: FAISSStore | None = None


def get_store() -> FAISSStore:
    global _store
    if _store is None:
        _store = FAISSStore()
        _store.load()
    return _store

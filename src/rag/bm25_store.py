"""BM25 lexical index, persisted alongside the FAISS vector index.

Maintains a pickled corpus of `(chunk_id_int, doc_id, tokens)` rows and a
serialised `bm25s.BM25` index. The index is **rebuilt** on every upsert
(corpus is small - government scheme docs, so re-tokenising is cheap),
which keeps the implementation simple and the on-disk state always
consistent with FAISS.
"""

from __future__ import annotations

import pickle  # nosec B403 - corpus is app-generated and never user-supplied
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import bm25s

from src.utils.config import bm25_corpus_path, bm25_index_dir

# Cheap multilingual tokenizer: lower-case, split on non-letters/digits.
# `\w` is Unicode-aware but drops combining marks, which destroys
# Devanagari/Bengali syllables. The explicit script ranges restore them.
_INDIC_RANGES = (
    r"ऀ-ॿ"  # Devanagari (letters + marks)
    r"ঀ-৿"  # Bengali
)
_TOKEN_RE = re.compile(rf"[\w{_INDIC_RANGES}]+", flags=re.UNICODE)

# Short, mostly-noise tokens. Kept tiny - BM25 itself down-weights frequent terms.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "but",
        "if",
        "with",
        "as",
        "by",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
    }
)


def tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


@dataclass
class BM25Hit:
    chunk_id_int: int
    doc_id: str
    score: float


class BM25Store:
    """In-memory BM25 index with disk persistence.

    On-disk layout::

        data/index/bm25_corpus.pkl   # pickled list[CorpusRow]
        data/index/bm25/             # bm25s native index directory
    """

    def __init__(
        self,
        index_dir: Path | None = None,
        corpus_path: Path | None = None,
    ) -> None:
        self.index_dir = index_dir or bm25_index_dir
        self.corpus_path = corpus_path or bm25_corpus_path
        # corpus: list of dicts {chunk_id_int, doc_id, tokens}
        self._corpus: list[dict[str, object]] = []
        self._bm25: bm25s.BM25 | None = None
        # Concurrent searchers may race the lazy rebuild; one builds, rest wait.
        self._rebuild_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self.corpus_path.exists():
            with open(self.corpus_path, "rb") as f:
                # nosec B301 - deserialising our own corpus file under data/index,
                # written by this app's ingestion path; never untrusted input.
                self._corpus = pickle.load(f)  # nosec B301
        else:
            self._corpus = []
        # Index is rebuilt from corpus on demand (cheap, deterministic)
        self._bm25 = None

    def save(self) -> None:
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.corpus_path, "wb") as f:
            pickle.dump(self._corpus, f)
        # Persist bm25s index too - load() rebuilds anyway, but having it on
        # disk makes the data dir self-describing.
        if self._bm25 is not None:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._bm25.save(str(self.index_dir))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def upsert(self, rows: list[tuple[int, str, str]]) -> None:
        """Add or replace rows.

        Args:
            rows: list of ``(chunk_id_int, doc_id, text_en)`` tuples.
        """
        if not rows:
            return
        incoming_ids = {cid for cid, _, _ in rows}
        self._corpus = [
            r for r in self._corpus if r["chunk_id_int"] not in incoming_ids
        ]
        for cid, doc_id, text in rows:
            self._corpus.append(
                {"chunk_id_int": cid, "doc_id": doc_id, "tokens": tokenize(text)}
            )
        self._bm25 = None  # mark dirty; rebuild on next search

    def remove_doc(self, doc_id: str) -> int:
        before = len(self._corpus)
        self._corpus = [r for r in self._corpus if r["doc_id"] != doc_id]
        removed = before - len(self._corpus)
        if removed:
            self._bm25 = None
        return removed

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _ensure_index(self) -> bm25s.BM25 | None:
        if not self._corpus:
            return None
        if self._bm25 is None:
            with self._rebuild_lock:
                if self._bm25 is None:
                    bm25 = bm25s.BM25()
                    bm25.index([row["tokens"] for row in self._corpus])
                    self._bm25 = bm25
        return self._bm25

    def search(self, query: str, top_n: int) -> list[BM25Hit]:
        bm25 = self._ensure_index()
        if bm25 is None:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        k = min(top_n, len(self._corpus))
        # bm25s returns (doc_indices, scores); we wrap into BM25Hit
        results, scores = bm25.retrieve([q_tokens], k=k)
        idxs = results[0].tolist()
        score_row = scores[0].tolist()
        hits: list[BM25Hit] = []
        for idx, score in zip(idxs, score_row):
            if score <= 0.0:
                continue  # bm25s returns top-k even when no term matches
            row = self._corpus[idx]
            hits.append(
                BM25Hit(
                    chunk_id_int=int(row["chunk_id_int"]),
                    doc_id=str(row["doc_id"]),
                    score=float(score),
                )
            )
        return hits

    @property
    def total_chunks(self) -> int:
        return len(self._corpus)


# Module-level singleton - loaded lazily, mirrors FAISSStore pattern
_store: BM25Store | None = None


def get_bm25_store() -> BM25Store:
    global _store
    if _store is None:
        _store = BM25Store()
        _store.load()
    return _store

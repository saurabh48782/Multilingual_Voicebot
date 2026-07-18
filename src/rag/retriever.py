"""Hybrid retriever: BM25 + dense vector → RRF fusion → cross-encoder rerank.

Pipeline:
  1. Embed query (e5 with "query:" prefix) → FAISS top-N candidates.
  2. Tokenise query → BM25 top-N candidates.
  3. Fuse the two ranked lists with Reciprocal Rank Fusion (RRF).
  4. Optionally rerank the fused candidates with a cross-encoder;
     score is sigmoid-scaled into (0, 1).
  5. Apply the dual-threshold confidence gate (top score + top1/top2 gap).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.bm25_store import BM25Store, get_bm25_store
from src.rag.embedder import embed_query
from src.rag.locks import index_rwlock
from src.rag.store import FAISSStore, SearchResult, get_store
from src.utils.config import cfg


@dataclass
class RetrievalResult:
    docs: list[SearchResult]
    top_score: float
    gap: float
    passed: bool


def confidence_gate(
    top: float,
    second: float,
    threshold: float,
    gap_threshold: float,
    absolute_scores: bool,
) -> bool:
    """Dual-threshold gate: absolute top score + top1/top2 separation.

    With absolute (reranker sigmoid) scores, a runner-up that also clears the
    threshold means the corpus holds redundant supporting chunks - that is
    corroboration, not ambiguity, so the gap requirement is waived. Normalised
    RRF scores are relative (top is always 1.0), so there the gap is the only
    meaningful signal and no waiver applies.
    """
    if top < threshold:
        return False
    if top - second >= gap_threshold:
        return True
    return absolute_scores and second >= threshold


def _rrf_fuse(rankings: list[list[int]], rrf_k: int) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists of chunk_id_int.

    score(d) = Σ_i 1 / (rrf_k + rank_i(d))

    Returns list of (chunk_id_int, fused_score) sorted desc.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class Retriever:
    def __init__(
        self,
        store: FAISSStore | None = None,
        bm25: BM25Store | None = None,
    ) -> None:
        self._store = store
        self._bm25 = bm25

    @property
    def store(self) -> FAISSStore:
        if self._store is None:
            self._store = get_store()
        return self._store

    @property
    def bm25(self) -> BM25Store:
        if self._bm25 is None:
            self._bm25 = get_bm25_store()
        return self._bm25

    # Public API
    def search(self, query: str, k: int | None = None) -> RetrievalResult:
        """Hybrid retrieve + rerank + confidence gate."""
        rag = cfg["rag"]
        retr = rag.get("retrieval", {})
        k = k or rag["top_k"]

        vec_n = int(retr.get("vector_top_n", max(k * 4, 20)))
        bm25_n = int(retr.get("bm25_top_n", max(k * 4, 20)))
        rrf_k = int(retr.get("rrf_k", 60))
        use_reranker = bool(retr.get("use_reranker", True))
        rerank_batch = int(retr.get("rerank_batch_size", 16))

        # 1+2. Dense + lexical candidates - read lock keeps a concurrent
        # document upload from mutating FAISS/BM25/metadata mid-search.
        qvec = embed_query(query)
        with index_rwlock.read():
            dense_hits = self.store.search(qvec, top_k=vec_n)
            dense_ranking = [_search_result_id(r) for r in dense_hits]
            meta_lookup: dict[int, SearchResult] = {_search_result_id(r): r for r in dense_hits}

            bm25_hits = self.bm25.search(query, top_n=bm25_n)
            bm25_ranking = [h.chunk_id_int for h in bm25_hits]

            # BM25 hits may include chunks not in dense top-N - fetch their metadata
            missing_ids = [cid for cid in bm25_ranking if cid not in meta_lookup]
            if missing_ids:
                for sr in _hydrate_meta(self.store, missing_ids):
                    meta_lookup[_search_result_id(sr)] = sr

        # 3. RRF fusion
        fused = _rrf_fuse([dense_ranking, bm25_ranking], rrf_k=rrf_k)
        if not fused:
            return RetrievalResult(docs=[], top_score=0.0, gap=0.0, passed=False)

        # 4. Rerank (or keep RRF order)
        candidates = [meta_lookup[cid] for cid, _ in fused[: max(k * 4, 20)] if cid in meta_lookup]

        if use_reranker and candidates:
            from src.rag.reranker import rerank

            scores = rerank(query, [c.text_en for c in candidates], batch_size=rerank_batch)
            scored = [_with_score(c, s) for c, s in zip(candidates, scores, strict=False)]
            scored.sort(key=lambda sr: sr.score, reverse=True)
            docs = scored[:k]
        else:
            # No reranker: use RRF position; carry a normalised RRF score.
            # Filter to fused ids with metadata *before* slicing to k, so a
            # missing-metadata id below the cut doesn't under-fill the top-k.
            top_rrf = fused[0][1] or 1.0
            cid_to_rrf = {cid: score / top_rrf for cid, score in fused}
            docs = [
                _with_score(meta_lookup[cid], cid_to_rrf[cid])
                for cid, _ in fused
                if cid in meta_lookup
            ][:k]

        # 5. Confidence gate. Reranker scores are absolute sigmoids; the
        # RRF-only branch yields relative scores (top normalised to 1.0), so
        # it gets its own optional thresholds instead of reusing ones tuned
        # for a different scale.
        if use_reranker:
            threshold = float(rag["retrieval_threshold"])
            gap_threshold = float(rag["retrieval_gap_threshold"])
        else:
            threshold = float(retr.get("rrf_threshold", rag["retrieval_threshold"]))
            gap_threshold = float(retr.get("rrf_gap_threshold", rag["retrieval_gap_threshold"]))

        top_score = docs[0].score if docs else 0.0
        second_score = docs[1].score if len(docs) > 1 else 0.0
        gap = top_score - second_score
        passed = confidence_gate(
            top_score,
            second_score,
            threshold=threshold,
            gap_threshold=gap_threshold,
            absolute_scores=use_reranker,
        )
        return RetrievalResult(docs=docs, top_score=top_score, gap=gap, passed=passed)

    @classmethod
    def load(cls) -> Retriever:
        """Eagerly load FAISS + BM25 stores."""
        return cls(store=get_store(), bm25=get_bm25_store())


# Helpers
def _search_result_id(sr: SearchResult) -> int:
    """Reconstruct chunk_id_int from hex chunk_id (stable across runs)."""
    return int(sr.chunk_id, 16) % (2**63)


def _with_score(sr: SearchResult, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=sr.chunk_id,
        doc_id=sr.doc_id,
        chunk_index=sr.chunk_index,
        text_en=sr.text_en,
        source=sr.source,
        page_num=sr.page_num,
        score=score,
        headings=sr.headings,
        content_type=sr.content_type,
    )


def _hydrate_meta(store: FAISSStore, ids: list[int]) -> list[SearchResult]:
    """Look up FAISS metadata by chunk_id_int without running a search."""
    meta = store._meta
    if meta.empty:
        return []
    rows = meta[meta["chunk_id_int"].isin(ids)]
    out: list[SearchResult] = []
    for _, row in rows.iterrows():
        out.append(
            SearchResult(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                chunk_index=int(row["chunk_index"]),
                text_en=row["text_en"],
                source=row["source"],
                page_num=int(row["page_num"]),
                score=0.0,
                headings=str(row.get("headings", "") or ""),
                content_type=str(row.get("content_type", "text") or "text"),
            )
        )
    return out

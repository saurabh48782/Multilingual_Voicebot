"""Unit tests for Reciprocal Rank Fusion helper."""

from src.rag.retriever import _rrf_fuse


def test_rrf_doc_in_both_lists_wins() -> None:
    # Doc 1 ranked #1 in dense, #1 in BM25 - beats single-list docs.
    fused = _rrf_fuse([[1, 2, 3], [1, 4, 5]], rrf_k=60)
    ids = [cid for cid, _ in fused]
    assert ids[0] == 1


def test_rrf_higher_rank_beats_lower() -> None:
    fused = _rrf_fuse([[1, 2, 3, 4]], rrf_k=60)
    scores = dict(fused)
    assert scores[1] > scores[2] > scores[3] > scores[4]


def test_rrf_empty_input_returns_empty() -> None:
    assert _rrf_fuse([], rrf_k=60) == []
    assert _rrf_fuse([[], []], rrf_k=60) == []


def test_rrf_score_formula() -> None:
    # Single list: score = 1/(k + rank); rank is 1-indexed inside the helper.
    fused = _rrf_fuse([[7]], rrf_k=60)
    assert fused == [(7, 1.0 / 61.0)]

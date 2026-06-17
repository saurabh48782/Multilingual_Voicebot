"""Unit tests for the BM25 lexical index."""

from pathlib import Path

from src.rag.bm25_store import BM25Store, tokenize


def test_tokenize_lowercase_strips_stopwords() -> None:
    toks = tokenize("The PM Kisan scheme is for FARMERS.")
    assert "pm" in toks
    assert "kisan" in toks
    assert "scheme" in toks
    assert "farmers" in toks
    assert "the" not in toks
    assert "is" not in toks
    assert "for" not in toks


def test_tokenize_preserves_devanagari() -> None:
    toks = tokenize("किसान योजना")
    assert "किसान" in toks
    assert "योजना" in toks


def test_upsert_and_search_returns_relevant_hit(tmp_path: Path) -> None:
    store = BM25Store(index_dir=tmp_path / "bm25", corpus_path=tmp_path / "corpus.pkl")
    store.upsert(
        [
            (1, "d1", "PM Kisan scheme provides six thousand rupees per year"),
            (2, "d2", "Old age pension for senior citizens"),
            (3, "d3", "Housing scheme for rural families"),
        ]
    )

    hits = store.search("kisan", top_n=3)
    assert hits, "expected at least one BM25 hit"
    assert hits[0].chunk_id_int == 1


def test_remove_doc_filters_corpus() -> None:
    store = BM25Store()
    store.upsert(
        [
            (1, "d1", "kisan scheme"),
            (2, "d1", "kisan benefits"),
            (3, "d2", "pension scheme"),
        ]
    )
    assert store.total_chunks == 3
    removed = store.remove_doc("d1")
    assert removed == 2
    assert store.total_chunks == 1
    assert store.search("kisan", top_n=5) == []


def test_upsert_overwrites_existing_ids() -> None:
    store = BM25Store()
    store.upsert([(1, "d1", "original text")])
    store.upsert([(1, "d1", "updated text body")])
    assert store.total_chunks == 1
    hits = store.search("updated", top_n=1)
    assert hits and hits[0].chunk_id_int == 1


def test_persist_and_reload(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.pkl"
    index_dir = tmp_path / "bm25"
    a = BM25Store(index_dir=index_dir, corpus_path=corpus_path)
    a.upsert([(1, "d1", "alpha beta gamma"), (2, "d2", "delta epsilon zeta")])
    a.search("alpha", top_n=1)  # forces index build so it gets saved
    a.save()

    b = BM25Store(index_dir=index_dir, corpus_path=corpus_path)
    b.load()
    assert b.total_chunks == 2
    hits = b.search("alpha", top_n=1)
    assert hits and hits[0].chunk_id_int == 1


def test_empty_store_returns_no_hits() -> None:
    store = BM25Store()
    assert store.search("anything", top_n=5) == []


def test_search_with_empty_query_returns_no_hits() -> None:
    store = BM25Store()
    store.upsert([(1, "d1", "some content")])
    assert store.search("the of and", top_n=5) == []  # only stopwords

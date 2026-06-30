from __future__ import annotations

from src.rag.store import SearchResult


def make_doc(text: str = "PM Kisan: ₹6000/year.", score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id="c1",
        doc_id="doc1",
        chunk_index=0,
        text_en=text,
        source="doc.pdf",
        page_num=1,
        score=score,
    )

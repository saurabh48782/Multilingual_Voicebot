"""Shared lock guarding the FAISS + BM25 indexes against concurrent mutation.

`Retriever.search` takes the read side; ingestion takes the write side, so a
live `/api/documents` upload can never mutate the index (FAISS remove/add,
metadata DataFrame swap, BM25 corpus rebuild) while request threads are
searching it.

Process-local only — it does NOT coordinate across uvicorn workers or a
separate ingest CLI process. The API must run single-worker (see
scripts/start_api.sh) unless retrieval is moved to an external store.
"""

from __future__ import annotations

from src.utils.rwlock import RWLock

index_rwlock = RWLock()

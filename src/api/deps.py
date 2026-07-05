"""FastAPI dependencies - single source of truth for graph + audio cache.

The compiled `StateGraph`, `PostgresSaver`, and `AudioCache` are built once in
the app lifespan and stashed on `app.state`. Each request pulls them via the
dependency callables in this module.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, Request

from src.api.audio_cache import AudioCache


def get_graph(request: Request) -> Any:
    return request.app.state.graph


def get_audio_cache(request: Request) -> AudioCache:
    cache: AudioCache = request.app.state.audio_cache
    return cache


def get_checkpointer(request: Request) -> Any:
    return request.app.state.checkpointer


def get_db_pool(request: Request) -> Any:
    """asyncpg pool for session metadata. Absent under the test bootstrap and
    in dev without Postgres; the session_store helpers treat ``None`` as a
    no-op so the chat path is unaffected."""
    return getattr(request.app.state, "db_pool", None)


GraphDep = Depends(get_graph)
AudioCacheDep = Depends(get_audio_cache)
CheckpointerDep = Depends(get_checkpointer)
DbPoolDep = Depends(get_db_pool)

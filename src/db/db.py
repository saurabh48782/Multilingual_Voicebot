"""Postgres lifecycle for the API.

Request-time access is via ``get_db_pool`` / ``get_checkpointer`` in
``src.api.deps`` (both read ``app.state``); this module only owns setup/teardown.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.utils.config import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def db_pool_lifespan(app: FastAPI) -> AsyncIterator[asyncpg.Pool]:
    """Open the asyncpg pool for chat-session metadata, stash it on
    ``app.state.db_pool``, and close it on exit."""

    pool = await asyncpg.create_pool(cfg["memory"]["checkpoint_dsn"], min_size=1, max_size=5)
    app.state.db_pool = pool
    try:
        yield pool
    finally:
        await pool.close()


@asynccontextmanager
async def checkpointer_lifespan(app: FastAPI) -> AsyncIterator[AsyncPostgresSaver]:
    """Open the LangGraph ``AsyncPostgresSaver`` (conversation content), run its
    one-time ``setup()``, and stash it on ``app.state.checkpointer``."""
    async with AsyncPostgresSaver.from_conn_string(cfg["memory"]["checkpoint_dsn"]) as checkpointer:
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        yield checkpointer

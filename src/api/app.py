from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.audio_cache import AudioCache
from src.api.routers import audio, chat, documents, evaluation, health, sessions, voice
from src.api.security import ApiKeyMiddleware, RateLimitMiddleware, api_key_configured
from src.api.templates import mount_frontend
from src.db.db import checkpointer_lifespan, db_pool_lifespan
from src.graph.builder import build_graph
from src.graph.deps import Deps
from src.rag.bm25_store import get_bm25_store
from src.rag.store import get_store
from src.utils.logger import get_logger, setup_logging
from src.utils.observability import configure_tracing

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    configure_tracing()

    if not api_key_configured():
        logger.warning(
            "VOICEBOT_API_KEY is not set - /api/* endpoints are UNAUTHENTICATED. "
            "Set it before exposing this service beyond localhost."
        )

    app.state.faiss_store = get_store()
    app.state.bm25_store = get_bm25_store()
    app.state.audio_cache = AudioCache()

    async with db_pool_lifespan(app), checkpointer_lifespan(app) as checkpointer:
        app.state.graph = build_graph(checkpointer=checkpointer, deps=Deps())
        logger.info("voicebot lifespan up - graph compiled, FAISS + BM25 loaded")
        yield


def create_app(bootstrap: Callable[[FastAPI], None] | None = None) -> FastAPI:
    """Build the app. `bootstrap` replaces the production lifespan in tests -
    it receives the app and must populate `app.state` itself."""
    if bootstrap is None:
        lifespan_ctx = lifespan
    else:

        @asynccontextmanager
        async def lifespan_ctx(app: FastAPI) -> AsyncIterator[None]:
            setup_logging()
            bootstrap(app)
            yield

    app = FastAPI(title="Multilingual Voicebot", lifespan=lifespan_ctx)

    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(RateLimitMiddleware)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # CORS only when explicitly configured; the bundled frontend is same-origin.
    origins = [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(health.router)
    app.include_router(voice.router)
    app.include_router(chat.router)
    app.include_router(sessions.router)
    app.include_router(audio.router)
    app.include_router(documents.router)
    app.include_router(evaluation.router)

    mount_frontend(app)
    return app


app = create_app()

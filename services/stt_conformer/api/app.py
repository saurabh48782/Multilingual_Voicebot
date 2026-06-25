"""FastAPI app factory + lifespan for the IndicConformer STT sidecar.

A lifespan that builds the shared ``Transcriber`` once and
stashes it on ``app.state`` for the request-scoped deps to pull.
The 600M model itself still loads lazily on the first /stt call,
so startup stays fast.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers import health, stt
from api.transcriber import Transcriber


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.transcriber = Transcriber()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="indic-conformer-stt", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(stt.router)
    return app

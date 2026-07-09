"""Shared fixtures: in-memory app + stubbed graph/audio cache."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import MemorySaver

from src.api.app import create_app
from src.api.audio_cache import AudioCache
from src.graph.builder import build_graph
from src.graph.deps import Deps
from tests.stubs import StubLLM, StubRetriever, StubSTT, StubTranslator, StubTTS

if TYPE_CHECKING:
    from src.rag.retriever import Retriever

# ---------- Fixtures ---------------------------------------------------------


@pytest.fixture
def audio_cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "audio_cache"
    d.mkdir()
    return d


def build_stub_app(cache_dir: Path, *, passed: bool = True, top_score: float = 0.9) -> FastAPI:
    """Build a fully-stubbed API app (no GPU / Ollama / Postgres).

    Single source of the test app-wiring, shared by the in-process httpx tests
    here and the live-uvicorn Tavern servers in ``routers/conftest.py``.
    ``passed`` / ``top_score`` steer the stub retriever so callers pick the
    confidence-pass (generate) or confidence-fail (fallback) branch.
    """
    checkpointer = MemorySaver()
    cache = AudioCache(cache_dir=cache_dir, ttl_seconds=3600, max_files=1000)
    deps = Deps(
        stt=StubSTT(),
        tts=StubTTS(),
        translator=StubTranslator(),
        llm=StubLLM(),
        retriever=cast("Retriever", StubRetriever(passed=passed, top_score=top_score)),
        audio_cache=cache,
    )
    graph = build_graph(checkpointer=checkpointer, deps=deps)

    def bootstrap(fastapi_app: FastAPI) -> None:
        fastapi_app.state.checkpointer = checkpointer
        fastapi_app.state.graph = graph
        fastapi_app.state.audio_cache = cache
        fastapi_app.state.faiss_store = None  # tests don't load FAISS

    return create_app(bootstrap=bootstrap)


@pytest.fixture
def app(audio_cache_dir: Path) -> FastAPI:
    return build_stub_app(audio_cache_dir)


@pytest_asyncio.fixture  # type: ignore[misc]  # untyped decorator
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

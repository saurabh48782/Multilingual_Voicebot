from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.security import ApiKeyMiddleware, RateLimitMiddleware


def _make_app(rate_limit: int | None = None) -> FastAPI:
    app = FastAPI()
    if rate_limit is not None:
        app.add_middleware(RateLimitMiddleware, limit_per_minute=rate_limit)
    app.add_middleware(ApiKeyMiddleware)

    @app.get("/api/ping")  # type: ignore[misc]
    def ping() -> dict[str, str]:
        return {"pong": "ok"}

    @app.get("/healthcheck")  # type: ignore[misc]
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return app


@pytest_asyncio.fixture  # type: ignore[misc]
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=_make_app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.parametrize(
    ("configured_key", "path", "sent_key", "expected"),
    [
        pytest.param(None, "/api/ping", None, 200, id="open-when-no-key-configured"),
        pytest.param("sekrit", "/api/ping", None, 401, id="missing-key-rejected"),
        pytest.param("sekrit", "/api/ping", "wrong", 401, id="wrong-key-rejected"),
        pytest.param("sekrit", "/api/ping", "sekrit", 200, id="correct-key-accepted"),
        pytest.param("sekrit", "/healthcheck", None, 200, id="healthcheck-stays-open"),
    ],
)
async def test_api_key_enforcement(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    configured_key: str | None,
    path: str,
    sent_key: str | None,
    expected: int,
) -> None:
    if configured_key is None:
        monkeypatch.delenv("VOICEBOT_API_KEY", raising=False)
    else:
        monkeypatch.setenv("VOICEBOT_API_KEY", configured_key)
    headers = {"X-API-Key": sent_key} if sent_key is not None else {}
    r = await client.get(path, headers=headers)
    assert r.status_code == expected


async def test_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOICEBOT_API_KEY", raising=False)
    transport = ASGITransport(app=_make_app(rate_limit=3))
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        statuses = [(await ac.get("/api/ping")).status_code for _ in range(5)]
    assert statuses[:3] == [200, 200, 200]
    assert 429 in statuses[3:]
    # Unprotected paths are never throttled.
    async with AsyncClient(
        transport=ASGITransport(app=_make_app(rate_limit=1)), base_url="http://test"
    ) as ac:
        await ac.get("/api/ping")
        r = await ac.get("/healthcheck")
    assert r.status_code == 200

"""Live stubbed servers for the Tavern YAML integration tests.

Tavern drives the API over real HTTP (via ``requests``), so unlike the
in-process ``httpx``/``ASGITransport`` tests it needs the app bound to an
actual socket. This conftest boots two uvicorn servers in background threads,
both wired with the deterministic stub providers from ``tests/stubs.py``
(no GPU / Ollama / Postgres):

* ``API_URL``      - happy stack (retriever passes, high confidence)
* ``FALLBACK_URL`` - low-confidence stack (retriever fails -> fallback path)

Both URLs are exported as environment variables so the YAML files can
reference ``{tavern.env_vars.API_URL}`` / ``{tavern.env_vars.FALLBACK_URL}``.
The autouse session fixture starts them once for the whole ``routers/`` suite.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from tests.integration.api.conftest import build_stub_app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _LiveServer:
    """A uvicorn server running in a daemon thread on a fixed local port."""

    def __init__(self, app: FastAPI) -> None:
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="warning", lifespan="on"
        )
        self._server = uvicorn.Server(config)
        # uvicorn skips signal-handler install off the main thread, so this is safe.
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self, timeout: float = 30.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server.started:
                try:
                    if httpx.get(f"{self.base_url}/healthcheck", timeout=2.0).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
            time.sleep(0.1)
        raise RuntimeError(f"uvicorn did not become ready on {self.base_url}")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10.0)


@pytest.fixture(scope="session", autouse=True)
def tavern_live_servers(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    # Snapshot every var this fixture touches so the whole test process's
    # environment is restored afterwards, not just the ones we always set.
    _keys = ("VOICEBOT_API_KEY", "RATE_LIMIT_PER_MINUTE", "API_URL", "FALLBACK_URL")
    _saved = {k: os.environ.get(k) for k in _keys}

    # Keep the suite deterministic: no API key gate, generous rate limit.
    os.environ.pop("VOICEBOT_API_KEY", None)
    os.environ["RATE_LIMIT_PER_MINUTE"] = "100000"

    happy = _LiveServer(
        build_stub_app(tmp_path_factory.mktemp("happy_cache"), passed=True, top_score=0.9)
    )
    fallback = _LiveServer(
        build_stub_app(tmp_path_factory.mktemp("fallback_cache"), passed=False, top_score=0.2)
    )
    happy.start()
    fallback.start()

    os.environ["API_URL"] = happy.base_url
    os.environ["FALLBACK_URL"] = fallback.base_url
    try:
        yield
    finally:
        happy.stop()
        fallback.stop()
        for key, value in _saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

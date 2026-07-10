"""API-key auth and per-IP rate limiting for the public API.

Both are env-driven so local dev works with zero config:

- ``VOICEBOT_API_KEY``: when set, every ``/api/*`` request must carry the same
  value in an ``X-API-Key`` header. Unset → auth disabled (a warning is logged
  at startup). ``/healthcheck``, ``/audio/*`` (unguessable UUID ids) and the
  static frontend stay open.
- ``RATE_LIMIT_PER_MINUTE``: sliding-window per-client-IP limit on ``/api/*``
  routes (default 120).
  When ``RATE_LIMIT_TRUST_FORWARDED`` is set the client IP is taken from the
  left-most ``X-Forwarded-For`` hop (set this only behind a trusted reverse proxy
  that overwrites the header - otherwise clients can spoof it to evade the limit).
- ``MAX_REQUEST_BYTES``: hard cap on request body size (default 50 MB) enforced
  by ``MaxBodySizeMiddleware`` *before* the body is buffered into memory, so an
  oversized upload is rejected with 413 rather than materialized in RAM.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROTECTED_PREFIX = "/api/"
_DEFAULT_MAX_REQUEST_BYTES = 50 * 1024 * 1024  # 50 MB


def api_key_configured() -> bool:
    return bool(os.environ.get("VOICEBOT_API_KEY"))


class ApiKeyMiddleware(BaseHTTPMiddleware):  # type: ignore[misc, unused-ignore]
    """Require X-API-Key on /api/* when VOICEBOT_API_KEY is set."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        expected = os.environ.get("VOICEBOT_API_KEY", "")
        if expected and request.url.path.startswith(_PROTECTED_PREFIX):
            provided = request.headers.get("X-API-Key", "")
            if not secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
                return JSONResponse({"detail": "invalid or missing API key"}, status_code=401)
        response: Response = await call_next(request)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):  # type: ignore[misc, unused-ignore]
    """Sliding-window per-IP limiter on /api/* routes."""

    def __init__(self, app: Any, limit_per_minute: int | None = None) -> None:
        super().__init__(app)
        self._limit = limit_per_minute or int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
        self._trust_forwarded = bool(os.environ.get("RATE_LIMIT_TRUST_FORWARDED"))
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _client_ip(self, request: Request) -> str:
        if self._trust_forwarded:
            forwarded = request.headers.get("X-Forwarded-For", "")
            if forwarded:
                # Left-most hop is the original client; the proxy appends itself.
                return forwarded.split(",")[0].strip()  # type: ignore[no-any-return, unused-ignore]
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if not request.url.path.startswith(_PROTECTED_PREFIX):
            passthrough: Response = await call_next(request)
            return passthrough

        client_ip = self._client_ip(request)
        now = time.monotonic()
        with self._lock:
            window = self._hits[client_ip]
            while window and now - window[0] > 60.0:
                window.popleft()
            if len(window) >= self._limit:
                return JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )
            window.append(now)
            # Bound memory: drop clients whose most-recent hit has aged out of the
            # window once the table grows large. Idle deques are never drained by
            # the loop above (that only touches the requesting IP), so purge by the
            # age of the newest entry rather than emptiness - otherwise stale IPs
            # (e.g. rotating IPv6 /64s) accumulate unbounded.
            if len(self._hits) > 10_000:
                stale = [ip for ip, q in self._hits.items() if not q or now - q[-1] > 60.0]
                for ip in stale:
                    del self._hits[ip]
        response: Response = await call_next(request)
        return response


class _BodyTooLargeError(Exception):
    """Raised from the wrapped receive when the streamed body exceeds the cap."""


class MaxBodySizeMiddleware:
    """Reject oversized request bodies before they are buffered into memory."""

    def __init__(self, app: ASGIApp, max_bytes: int | None = None) -> None:
        self.app = app
        self.max_bytes = max_bytes or int(
            os.environ.get("MAX_REQUEST_BYTES", str(_DEFAULT_MAX_REQUEST_BYTES))
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break
                if declared > self.max_bytes:
                    await self._send_413(send)
                    return
                break

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLargeError:
            if response_started:
                # Body overran mid-response - too late to send a clean 413.
                raise
            await self._send_413(send)

    async def _send_413(self, send: Send) -> None:
        body = b'{"detail":"request body too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

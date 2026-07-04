"""API-key auth and per-IP rate limiting for the public API.

Both are env-driven so local dev works with zero config:

- ``VOICEBOT_API_KEY``: when set, every ``/api/*`` request must carry the same
  value in an ``X-API-Key`` header. Unset → auth disabled (a warning is logged
  at startup). ``/healthcheck``, ``/audio/*`` (unguessable UUID ids) and the
  static frontend stay open.
- ``RATE_LIMIT_PER_MINUTE``: sliding-window per-client-IP limit on ``/api/*``
  routes (default 120). In-process only - adequate for the single-worker
  deployment this app requires.
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

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROTECTED_PREFIX = "/api/"


def api_key_configured() -> bool:
    return bool(os.environ.get("VOICEBOT_API_KEY"))


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Require X-API-Key on /api/* when VOICEBOT_API_KEY is set."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        expected = os.environ.get("VOICEBOT_API_KEY", "")
        if expected and request.url.path.startswith(_PROTECTED_PREFIX):
            provided = request.headers.get("X-API-Key", "")
            if not secrets.compare_digest(provided, expected):
                return JSONResponse({"detail": "invalid or missing API key"}, status_code=401)
        response: Response = await call_next(request)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window per-IP limiter on /api/* routes."""

    def __init__(self, app: Any, limit_per_minute: int | None = None) -> None:
        super().__init__(app)
        self._limit = limit_per_minute or int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if not request.url.path.startswith(_PROTECTED_PREFIX):
            passthrough: Response = await call_next(request)
            return passthrough

        client_ip = request.client.host if request.client else "unknown"
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
            # Bound memory: drop idle clients once the table grows large.
            if len(self._hits) > 10_000:
                stale = [ip for ip, q in self._hits.items() if not q]
                for ip in stale:
                    del self._hits[ip]
        response: Response = await call_next(request)
        return response

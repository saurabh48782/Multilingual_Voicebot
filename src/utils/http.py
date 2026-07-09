"""Shared synchronous HTTP client for the sidecar / Ollama calls.

A single module-level ``httpx.Client`` keeps a connection pool alive across
requests (keep-alive to the STT/TTS sidecars and Ollama) instead of paying a
fresh TCP handshake per call. ``post_with_retry`` adds exponential backoff on
transient *transport* errors (DNS, connection refused/reset, read timeout);
HTTP error *responses* are returned untouched so the caller decides via
``raise_for_status``. Retrying is safe here because a transport error means the
request did not complete server-side.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Generous read window: the first sidecar/Ollama call also loads the model.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)

_client: httpx.Client | None = None
_client_lock = threading.Lock()


def get_client() -> httpx.Client:
    """Return the shared client, creating it on first use (thread-safe)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(timeout=_DEFAULT_TIMEOUT)
    return _client


def post_with_retry(
    url: str,
    *,
    retries: int = 2,
    backoff: float = 0.5,
    **kwargs: Any,
) -> httpx.Response:
    """POST via the shared client, retrying transport errors with backoff."""
    client = get_client()
    last_exc: httpx.TransportError | None = None
    for attempt in range(retries + 1):
        try:
            return client.post(url, **kwargs)
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt == retries:
                break
            sleep_s = backoff * (2**attempt)
            logger.warning(
                "POST %s failed (%s); retry %d/%d in %.1fs",
                url,
                type(exc).__name__,
                attempt + 1,
                retries,
                sleep_s,
            )
            time.sleep(sleep_s)
    if last_exc is None:  # unreachable, but keeps the type checker honest
        raise RuntimeError(f"POST {url} failed without an exception")
    raise last_exc

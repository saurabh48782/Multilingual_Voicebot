"""Groq-hosted LLM client"""

from __future__ import annotations

from groq import Groq

from src.llm.base import LLMResponse
from src.utils.config import cfg

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = cfg["api"]["groq_api_key"]
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set - add it to .env")
        # SDK retries 429/5xx/connection errors with exponential backoff.
        _client = Groq(api_key=key, timeout=30.0, max_retries=2)
    return _client


class GroqLLM:
    """Thin wrapper around Groq chat completions."""

    def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        json_mode: bool = False,
        think: bool = False,  # Groq models have no thinking toggle; accepted for protocol parity.
    ) -> LLMResponse:
        model = model or cfg["llm"]["model"]
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = _get_client().chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=resp.model,
            usage=resp.usage.model_dump() if resp.usage else {},
        )

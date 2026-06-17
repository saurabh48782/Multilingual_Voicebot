"""Ollama local LLM client"""

from __future__ import annotations

import httpx

from src.llm.base import LLMResponse
from src.utils.config import cfg


class OllamaLLM:
    """Calls Ollama's /api/chat endpoint synchronously."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self._base_url = (base_url or cfg["llm"]["ollama_base_url"]).rstrip("/")
        self._default_model = model or cfg["llm"]["ollama_model"]

    def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        json_mode: bool = False,
        think: bool = False,
    ) -> LLMResponse:
        model = model or self._default_model
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        if think:
            # Native Ollama thinking: reasoning is returned in message.thinking,
            # keeping message.content as the clean final answer.
            payload["think"] = True

        resp: httpx.Response | None = None
        for attempt in range(2):
            try:
                resp = httpx.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                    timeout=120.0,
                )
                break
            except httpx.TransportError:
                if attempt == 1:
                    raise
        if resp is None:
            raise RuntimeError("Ollama request failed without a response")
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return LLMResponse(
            content=content,
            model=model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        )

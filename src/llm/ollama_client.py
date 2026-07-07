"""Ollama local LLM client"""

from __future__ import annotations

from typing import Any

import httpx

from src.llm.base import LLMResponse
from src.utils.config import cfg
from src.utils.observability import llm_run_outputs, strip_self, traceable


class OllamaLLM:
    """Calls Ollama's /api/chat endpoint synchronously."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self._base_url = (base_url or cfg["llm"]["ollama_base_url"]).rstrip("/")
        self._default_model = model or cfg["llm"]["model"]

    # LangGraph traces the node span; this makes the actual LLM call inside it
    # a nested llm-type span with prompt/completion token usage.
    @traceable(  # type: ignore[misc]
        run_type="llm",
        name="ollama_chat",
        process_inputs=strip_self,
        process_outputs=llm_run_outputs,
    )
    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        json_mode: bool = False,
        think: bool = False,
        temperature: float | None = None,
        num_ctx: int | None = None,
    ) -> LLMResponse:
        model = model or self._default_model
        # Always pin num_ctx: Ollama's default window (2048/4096) is smaller than
        # our system prompt + retrieved context, and it silently truncates from
        # the FRONT - dropping the system prompt (injection guard, INSUFFICIENT_
        # CONTEXT rule) before the model ever sees it. temperature is per-call:
        # deterministic nodes (verify/translate/rewrite) pass 0.0.
        options: dict[str, Any] = {
            "num_ctx": num_ctx if num_ctx is not None else cfg["llm"].get("num_ctx", 8192),
        }
        if temperature is not None:
            options["temperature"] = temperature
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
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

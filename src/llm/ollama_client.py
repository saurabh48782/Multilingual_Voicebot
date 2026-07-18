"""Ollama local LLM client"""

from __future__ import annotations

import json
from typing import Any

from src.llm.base import LLMResponse
from src.utils.config import cfg
from src.utils.http import post_with_retry, stream_lines
from src.utils.observability import llm_run_outputs, strip_self, traceable


class OllamaLLM:
    """Calls Ollama's /api/chat endpoint synchronously."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self._base_url = (base_url or cfg["llm"]["ollama_base_url"]).rstrip("/")
        self._default_model = model or cfg["llm"]["model"]

    # LangGraph traces the node span; this makes the actual LLM call inside it
    # a nested llm-type span with prompt/completion token usage.
    @traceable(
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
        stream: bool = False,
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
            "stream": stream,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"
        if think:
            # Native Ollama thinking: reasoning is returned in message.thinking,
            # keeping message.content as the clean final answer.
            payload["think"] = True

        endpoint = f"{self._base_url}/api/chat"
        if stream:
            return self._complete_streaming(endpoint, payload, model)

        # Shared pooled client + exponential backoff on transient transport errors.
        resp = post_with_retry(endpoint, json=payload)
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

    def _complete_streaming(
        self, endpoint: str, payload: dict[str, Any], model: str
    ) -> LLMResponse:
        """Consume Ollama's NDJSON stream, reassembling the full reply.

        Streaming is a timeout strategy, not a UX one: the caller still gets one
        complete ``LLMResponse``. Because bytes arrive continuously the httpx
        ``read`` timeout bounds the inter-token gap instead of the whole
        generation, so a long reply (e.g. summarising a full conversation) no
        longer trips a ReadTimeout. Only ``message.content`` deltas are kept;
        ``message.thinking`` deltas (think mode) are ignored just like the
        non-streaming path strips them. Usage counts arrive on the final
        ``done`` object.
        """
        content_parts: list[str] = []
        usage: dict[str, Any] = {}
        for line in stream_lines(endpoint, json=payload):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate the occasional keep-alive / blank frame
            if error := obj.get("error"):
                raise RuntimeError(f"Ollama stream error: {error}")
            piece = obj.get("message", {}).get("content")
            if piece:
                content_parts.append(piece)
            if obj.get("done"):
                usage = {
                    "prompt_tokens": obj.get("prompt_eval_count", 0),
                    "completion_tokens": obj.get("eval_count", 0),
                }
        return LLMResponse(content="".join(content_parts), model=model, usage=usage)

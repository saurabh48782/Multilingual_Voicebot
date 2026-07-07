"""LLM provider protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        json_mode: bool = False,
        think: bool = False,
        temperature: float | None = None,
        num_ctx: int | None = None,
    ) -> LLMResponse: ...

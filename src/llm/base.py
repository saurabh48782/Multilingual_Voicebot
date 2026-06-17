"""LLM provider protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[dict],
        model: str | None = None,
        json_mode: bool = False,
        think: bool = False,
    ) -> LLMResponse: ...

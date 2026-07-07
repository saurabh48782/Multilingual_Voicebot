"""Translation provider protocol and shared types."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.llm.base import LLMProvider

SUPPORTED_VERNACULARS: frozenset[str] = frozenset({"hi", "bn"})
SUPPORTED_LANGUAGES: frozenset[str] = SUPPORTED_VERNACULARS | {"en"}


@dataclass
class TranslationResult:
    text: str
    source_language: str
    target_language: str


@runtime_checkable
class TranslationProvider(Protocol):
    def to_english(self, text: str, source_language: str) -> TranslationResult: ...

    def to_vernacular(self, text: str, target_language: str) -> TranslationResult: ...

    def to_english_batch(self, texts: list[str]) -> list[str]: ...


# ---------------------------------------------------------------------------
# Shared batch translation (used by corpus ingestion)
#
# Source-language-agnostic: ingestion runs before language detection and a
# single document may mix scripts, so the prompt instructs the model to copy
# already-English passages unchanged rather than relying on a source language.
# Passages are numbered and batched to cut the round-trip count on large
# corpora (~10x fewer LLM calls than one request per passage).
# ---------------------------------------------------------------------------

_BATCH_SIZE = 10
_NUMBERED_LINE_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")


def _parse_numbered(raw: str, expected: int, fallback: list[str]) -> list[str]:
    """Extract [N]-prefixed passages from a numbered translation response.

    A passage spans from its [N] marker up to the next marker, so multi-line
    translations are kept whole instead of being truncated to their first line.
    Missing entries fall back to the original passage text.
    """
    entries: dict[int, list[str]] = {}
    current: int | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        match = _NUMBERED_LINE_RE.match(stripped)
        if match:
            current = int(match.group(1))
            entries[current] = [match.group(2)]
        elif current is not None and stripped and stripped != "---":
            entries[current].append(stripped)

    out: list[str] = []
    for i in range(expected):
        text = " ".join(entries.get(i + 1, [])).strip()
        out.append(text if text else fallback[i])
    return out


def translate_batch_to_english(llm: LLMProvider, model: str, texts: list[str]) -> list[str]:
    """Translate passages to English in numbered batches via the given LLM.

    Shared by every translation provider's ``to_english_batch``; returns a list
    aligned with ``texts`` (already-English passages pass through unchanged).
    """
    results: list[str] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        numbered = "\n---\n".join(f"[{j + 1}] {t}" for j, t in enumerate(batch))
        prompt = (
            "Translate the following numbered passages to English. "
            "Preserve numbering. Output only the translated passages, one per line, "
            "keeping the [N] prefix. If a passage is already in English, copy it unchanged.\n\n"
            + numbered
        )
        resp = llm.complete(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,  # faithful, deterministic translation
        )
        results.extend(_parse_numbered(resp.content or "", len(batch), batch))
    return results

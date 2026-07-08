"""Translation provider protocol and shared types."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.llm.base import LLMProvider

logger = get_logger(__name__)

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

# Devanagari (Hindi) + Bengali script ranges. The English-only index must not be
# fed vernacular text: a failed/echoed translation still carries these glyphs.
_INDIC_SCRIPT_RE = re.compile(r"[ऀ-ॿঀ-৿]")


def _looks_untranslated(text: str) -> bool:
    """True when `text` still contains a meaningful share of Indic script,
    i.e. the translation to English failed or the model echoed the original.

    Used as an ingest-time gate so vernacular text never silently lands in the
    English-only vector index (it degrades cross-lingual retrieval quality)."""
    if not text:
        return False
    indic = len(_INDIC_SCRIPT_RE.findall(text))
    # A stray transliterated proper noun is fine; a passage that's mostly Indic
    # glyphs is an untranslated one.
    return indic >= 5 and indic / max(len(text), 1) > 0.10


def _translate_one(llm: LLMProvider, model: str, text: str) -> str:
    resp = llm.complete(
        messages=[
            {
                "role": "user",
                "content": (
                    "Translate the following passage to English. Output only the "
                    "translation, nothing else. If it is already English, copy it "
                    "unchanged.\n\n" + text
                ),
            }
        ],
        model=model,
        temperature=0.0,
    )
    return (resp.content or "").strip()


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
        parsed = _parse_numbered(resp.content or "", len(batch), batch)

        # Gate: any passage that came back still in Indic script is a failed
        # batch translation (dropped entry → vernacular fallback, or the model
        # echoed the original). Retry it once on its own; if it still fails,
        # log it — better a warned gap than poisoning the English index.
        for idx, translated_text in enumerate(parsed):
            if not _looks_untranslated(translated_text):
                continue
            try:
                retry = _translate_one(llm, model, batch[idx])
            except Exception:
                logger.warning("Single-passage translation retry raised; keeping original")
                retry = ""
            if retry and not _looks_untranslated(retry):
                parsed[idx] = retry
            else:
                logger.warning(
                    "Passage still not English after retry - indexing may be degraded "
                    "(passage prefix: %s)",
                    batch[idx][:60],
                )
        results.extend(parsed)
    return results

"""Indic Parler-TTS provider: HTTP client to the services/tts_parler sidecar.

The voice description is resolved app-side from params.yaml (single source of
voice config) and sent per request. See services/tts_parler/README.md for why TTS
runs in an isolated sidecar.

Synchronous on purpose: synthesize() runs in a LangGraph worker thread (sync node).
"""

from __future__ import annotations

import httpx

from src.utils.config import cfg
from src.utils.observability import redact_audio_outputs, strip_self, traceable

# Synthesis can take several seconds (esp. CPU); first call also loads the model.
_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)

# Generic voice description; works for every supported language. Naming a
# recommended speaker (e.g. "Divya" for Hindi) yields a more consistent voice.
# Overridable globally via tts.description or per-language via tts.descriptions.
_DEFAULT_DESCRIPTION = (
    "A clear, calm voice speaks at a natural, moderate pace in a quiet "
    "environment. The recording is very high quality, with the voice sounding "
    "close-up and natural, with no background noise."
)


def _description_for(language: str) -> str:
    tts_cfg = cfg.get("tts", {})
    per_lang = tts_cfg.get("descriptions", {}) or {}
    return per_lang.get(language) or tts_cfg.get("description") or _DEFAULT_DESCRIPTION


class IndicParlerRemoteTts:
    """Synthesize speech via the indic-parler-tts sidecar over HTTP."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    # Traced as a tool span; audio output is redacted to a byte count.
    @traceable(
        run_type="tool",
        name="tts_sidecar",
        process_inputs=strip_self,
        process_outputs=redact_audio_outputs,
    )
    def synthesize(self, text: str, language: str) -> bytes:
        resp = httpx.post(
            f"{self._base_url}/tts",
            json={"text": text, "description": _description_for(language)},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.content

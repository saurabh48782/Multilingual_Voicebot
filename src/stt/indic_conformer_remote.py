"""IndicConformer STT provider: HTTP client to the services/stt_conformer sidecar.

decode_strategy is resolved app-side from params.yaml (single source of STT
config) and sent per request. See services/stt_conformer/README.md for why STT
runs in an isolated sidecar.
"""

from __future__ import annotations

from src.stt.base import TranscriptionResult
from src.utils.config import cfg
from src.utils.http import post_with_retry
from src.utils.observability import redact_audio_inputs, traceable


class IndicConformerRemoteStt:
    """Transcribe speech via the indic-conformer sidecar over HTTP."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    # Traced as a tool span; raw audio bytes are redacted to a byte count.
    @traceable(run_type="tool", name="stt_sidecar", process_inputs=redact_audio_inputs)
    def transcribe(self, audio: bytes, language: str) -> TranscriptionResult:
        lang = language.lower()
        strategy = (cfg.get("stt", {}).get("decode_strategy") or "rnnt").lower()
        resp = post_with_retry(
            f"{self._base_url}/stt",
            files={"audio": ("audio", audio, "application/octet-stream")},
            data={"language": lang, "decode_strategy": strategy},
        )
        resp.raise_for_status()
        body = resp.json()
        return TranscriptionResult(
            text=(body.get("text") or "").strip(),
            language=(body.get("language") or lang),
        )

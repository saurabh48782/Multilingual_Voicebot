"""Unit tests for the IndicConformer STT wiring.

Covers the remote HTTP client (request shape + response parsing) and the
_default_stt() provider-selection logic. STT runs only via the sidecar, so
_default_stt() requires stt.remote_url and raises otherwise. The remote client is
exercised with a stubbed httpx.post; the model itself is never loaded.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.graph import deps as deps_module
from src.stt import indic_conformer_remote as remote_module
from src.stt.base import STTProvider, TranscriptionResult
from src.stt.indic_conformer_remote import IndicConformerRemoteStt


class _FakeResponse:
    def __init__(self, payload: dict[str, str]) -> None:
        self._payload = payload
        self.raised = False

    def raise_for_status(self) -> None:
        self.raised = True

    def json(self) -> dict[str, str]:
        return self._payload


def test_remote_client_sends_multipart_and_parses_response(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["files"] = kwargs.get("files")
        captured["data"] = kwargs.get("data")
        return _FakeResponse({"text": "  नमस्ते  ", "language": "hi"})

    monkeypatch.setattr(remote_module.httpx, "post", fake_post)
    # decode_strategy is resolved app-side from cfg
    monkeypatch.setitem(deps_module.cfg["stt"], "decode_strategy", "ctc")

    stt = IndicConformerRemoteStt("http://stt:8002/")
    result = stt.transcribe(b"AUDIOBYTES", language="HI")

    assert isinstance(result, TranscriptionResult)
    assert result.text == "नमस्ते"  # stripped
    assert result.language == "hi"
    assert captured["url"] == "http://stt:8002/stt"  # trailing slash trimmed
    assert captured["data"] == {"language": "hi", "decode_strategy": "ctc"}
    assert "audio" in captured["files"]


def test_remote_client_normalises_language_to_lowercase(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        captured["data"] = kwargs.get("data")
        return _FakeResponse({"text": "x", "language": "bn"})

    monkeypatch.setattr(remote_module.httpx, "post", fake_post)

    IndicConformerRemoteStt("http://stt:8002").transcribe(b"a", language="BN")
    assert captured["data"]["language"] == "bn"


def test_default_stt_routes_remote_when_url_set(monkeypatch: Any) -> None:
    monkeypatch.setitem(deps_module.cfg["stt"], "remote_url", "http://stt:8002")
    provider = deps_module._default_stt()
    assert isinstance(provider, IndicConformerRemoteStt)
    assert isinstance(provider, STTProvider)  # satisfies the runtime-checkable Protocol


def test_default_stt_raises_when_url_unset(monkeypatch: Any) -> None:
    # No in-process backend: the sidecar URL is mandatory.
    monkeypatch.setitem(deps_module.cfg["stt"], "remote_url", "")
    with pytest.raises(RuntimeError):
        deps_module._default_stt()


def test_default_stt_rejects_unknown_provider(monkeypatch: Any) -> None:
    monkeypatch.setitem(deps_module.cfg["stt"], "provider", "whisper")
    with pytest.raises(ValueError):
        deps_module._default_stt()

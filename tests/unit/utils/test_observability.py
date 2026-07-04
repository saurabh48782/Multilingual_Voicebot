from __future__ import annotations

import os
from typing import Any

import pytest

from src.utils import observability as obs

_ENV_KEYS = (
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_ENDPOINT",
    "LANGSMITH_ENDPOINT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _patch_cfg(monkeypatch: pytest.MonkeyPatch, langsmith: dict[str, Any]) -> None:
    monkeypatch.setattr(obs, "cfg", {"observability": {"langsmith": langsmith}})


# configure_tracing
@pytest.mark.parametrize(
    ("langsmith_cfg", "pre_env", "expected", "expected_env"),
    [
        pytest.param(None, {}, False, {}, id="no-observability-section"),
        pytest.param(
            {"enabled": "false"},
            {"LANGCHAIN_TRACING_V2": "true"},  # stray env must not enable tracing
            False,
            {},
            id="disabled-forces-stray-env-off",
        ),
        pytest.param(
            {"enabled": "true", "api_key": ""},
            {},
            False,
            {},
            id="enabled-without-api-key",
        ),
        pytest.param(
            {"enabled": "true", "api_key": "ls-test-key", "project": "voicebot-dev"},
            {},
            True,
            {
                "LANGSMITH_API_KEY": "ls-test-key",
                "LANGSMITH_PROJECT": "voicebot-dev",
                "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",  # code default
            },
            id="enabled-with-api-key",
        ),
    ],
)
def test_configure_tracing(
    monkeypatch: pytest.MonkeyPatch,
    langsmith_cfg: dict[str, Any] | None,
    pre_env: dict[str, str],
    expected: bool,
    expected_env: dict[str, str],
) -> None:
    for key, value in pre_env.items():
        monkeypatch.setenv(key, value)
    if langsmith_cfg is None:
        monkeypatch.setattr(obs, "cfg", {})
    else:
        _patch_cfg(monkeypatch, langsmith_cfg)

    assert obs.configure_tracing() is expected
    assert obs.tracing_enabled() is expected
    for key, value in expected_env.items():
        assert os.environ[key] == value


# `enabled` is a "${LANGSMITH_TRACING:-false}" string or a YAML bool; only "true"
# counts. The YAML-bool case (True -> "True") is why the check lowercases.
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        ("true", True),
        ("True", True),
        ("TRUE", True),
        (False, False),
        ("false", False),
        ("", False),
    ],
)
def test_enabled_flag_parsing(monkeypatch: pytest.MonkeyPatch, value: Any, expected: bool) -> None:
    _patch_cfg(monkeypatch, {"enabled": value, "api_key": "k"})
    assert obs.configure_tracing() is expected


# graph_run_config
def test_graph_run_config_shape() -> None:
    config = obs.graph_run_config("sess-1", metadata={"language": "hi"})
    assert config["configurable"] == {"thread_id": "sess-1"}
    assert config["run_name"] == "voicebot_turn"
    assert "voicebot" in config["tags"]
    assert config["metadata"] == {"session_id": "sess-1", "language": "hi"}


def test_graph_run_config_drops_none_metadata() -> None:
    config = obs.graph_run_config("s", metadata={"language": None, "input_mode": "text"})
    assert config["metadata"] == {"session_id": "s", "input_mode": "text"}


# trace payload helpers
def test_strip_self() -> None:
    assert obs.strip_self({"self": object(), "text": "hi"}) == {"text": "hi"}


def test_llm_run_outputs_maps_usage() -> None:
    from src.llm.base import LLMResponse

    resp = LLMResponse(
        content="answer",
        model="gemma4:12b",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )
    out = obs.llm_run_outputs(resp)
    assert out["content"] == "answer"
    assert out["usage_metadata"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_redact_audio_inputs() -> None:
    cleaned = obs.redact_audio_inputs({"self": object(), "audio": b"\x00" * 128, "language": "hi"})
    assert cleaned["audio"] == {"audio_bytes": 128}
    assert cleaned["language"] == "hi"
    assert "self" not in cleaned


def test_redact_audio_outputs() -> None:
    assert obs.redact_audio_outputs(b"\x00" * 64) == {"audio_bytes": 64}
    assert obs.redact_audio_outputs("text") == {"output": "text"}

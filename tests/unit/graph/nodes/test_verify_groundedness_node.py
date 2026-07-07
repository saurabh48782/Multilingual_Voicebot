from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.graph.nodes.verify_groundedness import (
    make_verify_groundedness,
    parse_verdict,
    route_after_verify,
)
from src.graph.state import VoicebotState
from src.utils.config import cfg
from tests.unit.graph.nodes.conftest import make_doc


def _node(
    llm_content: str | None = None, side_effect: Exception | None = None
) -> Callable[[VoicebotState], dict[str, Any]]:
    deps = MagicMock()
    resp = MagicMock()
    if side_effect:
        deps.llm.complete.side_effect = side_effect
    else:
        resp.content = llm_content or ""
        deps.llm.complete.return_value = resp
    return make_verify_groundedness(deps)


# parse_verdict
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"grounded": true}', True),
        ('{"grounded": false}', False),
        ('{"grounded": true, "reasoning": "all claims supported"}', True),
        ('{"grounded": false, "reasoning": "claim not in context"}', False),
        ('```json\n{"grounded": true}\n```', True),
        ('```\n{"grounded": false}\n```', False),
    ],
    ids=[
        "true",
        "false",
        "true-with-reasoning",
        "false-with-reasoning",
        "json-fenced",
        "plain-fenced",
    ],
)
def test_parse_verdict_valid_json(content: str, expected: bool) -> None:
    assert parse_verdict(content) is expected


@pytest.mark.parametrize(
    "bad_content",
    [
        "not json at all",
        '{"grounded": true',
        "{}",
        "",
    ],
    ids=["plain-text", "truncated-json", "missing-key", "empty"],
)
def test_parse_verdict_malformed_raises(bad_content: str) -> None:
    with pytest.raises((json.JSONDecodeError, ValidationError)):
        parse_verdict(bad_content)


# route_after_verify
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"fallback_triggered": True}, "fallback"),
        ({"fallback_triggered": False}, "translate_to_vernacular"),
        ({}, "translate_to_vernacular"),
    ],
    ids=["fallback-triggered", "not-triggered", "key-absent"],
)
def test_route_after_verify(state: dict[str, Any], expected: str) -> None:
    assert route_after_verify(state) == expected  # type: ignore[arg-type]


# make_verify_groundedness: early exits
@pytest.mark.parametrize(
    "state",
    [
        {"fallback_triggered": True, "english_response": "Some answer."},
        {"fallback_triggered": False, "english_response": ""},
        {"retrieved_docs": [make_doc()]},
    ],
    ids=[
        "already-fallback-triggered",
        "empty-english-response",
        "missing-english-response",
    ],
)
def test_early_exit_skips_llm(state: dict[str, Any]) -> None:
    deps = MagicMock()
    node = make_verify_groundedness(deps)
    result = node(state)  # type: ignore[arg-type]
    deps.llm.complete.assert_not_called()
    assert result == {"grounded": False}


# make_verify_groundedness: LLM paths
@pytest.mark.parametrize(
    ("llm_content", "amount", "expected"),
    [
        ('{"grounded": true}', "₹6000", {"grounded": True}),
        (
            '{"grounded": false}',
            "₹9000",
            {
                "grounded": False,
                "fallback_triggered": True,
                "fallback_reason": "ungrounded",
            },
        ),
    ],
    ids=["grounded-true", "grounded-false-triggers-fallback"],
)
def test_grounded_verdict(llm_content: str, amount: str, expected: dict[str, Any]) -> None:
    node = _node(llm_content)
    state = {
        "english_response": f"PM Kisan gives {amount} per year.",
        "retrieved_docs": [make_doc()],
    }
    result = node(state)  # type: ignore[arg-type]
    assert result == expected


@pytest.mark.parametrize(
    "exc",
    [RuntimeError("provider down"), OSError("timeout"), Exception("boom")],
)
def test_llm_exception_fails_closed_with_verifier_error(exc: Exception) -> None:
    node = _node(side_effect=exc)
    state = {
        "english_response": "Some answer.",
        "retrieved_docs": [make_doc()],
    }
    result = node(state)  # type: ignore[arg-type]
    assert result == {
        "grounded": False,
        "fallback_triggered": True,
        "fallback_reason": "verifier_error",
    }


def test_malformed_llm_response_fails_closed() -> None:
    node = _node("not valid json")
    state = {
        "english_response": "Some answer.",
        "retrieved_docs": [make_doc()],
    }
    result = node(state)  # type: ignore[arg-type]
    assert result["grounded"] is False
    assert result["fallback_triggered"] is True


def test_llm_called_with_json_mode() -> None:
    deps = MagicMock()
    resp = MagicMock()
    resp.content = '{"grounded": true}'
    deps.llm.complete.return_value = resp
    node = make_verify_groundedness(deps)
    node({"english_response": "Answer.", "retrieved_docs": [make_doc()]})
    call_kwargs = deps.llm.complete.call_args.kwargs
    assert call_kwargs.get("json_mode") is True


def test_prompt_includes_the_question() -> None:
    deps = MagicMock()
    resp = MagicMock()
    resp.content = '{"grounded": true}'
    deps.llm.complete.return_value = resp
    node = make_verify_groundedness(deps)
    node(
        {
            "english_response": "Answer.",
            "retrieved_docs": [make_doc()],
            "rewritten_query": "What is the eligibility for PM Kisan?",
        }
    )
    call_kwargs = deps.llm.complete.call_args.kwargs
    user_content = call_kwargs["messages"][1]["content"]
    assert "What is the eligibility for PM Kisan?" in user_content
    assert "<question>" in user_content


def test_verifier_uses_generation_model_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(cfg["llm"], "verifier_model", None)
    deps = MagicMock()
    resp = MagicMock()
    resp.content = '{"grounded": true}'
    deps.llm.complete.return_value = resp
    node = make_verify_groundedness(deps)
    node({"english_response": "Answer.", "retrieved_docs": [make_doc()]})
    call_kwargs = deps.llm.complete.call_args.kwargs
    assert call_kwargs["model"] == cfg["llm"]["model"]


def test_verifier_model_override_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(cfg["llm"], "verifier_model", "different-family:7b")
    deps = MagicMock()
    resp = MagicMock()
    resp.content = '{"grounded": true}'
    deps.llm.complete.return_value = resp
    node = make_verify_groundedness(deps)
    node({"english_response": "Answer.", "retrieved_docs": [make_doc()]})
    call_kwargs = deps.llm.complete.call_args.kwargs
    assert call_kwargs["model"] == "different-family:7b"

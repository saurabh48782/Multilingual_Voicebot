"""Unit tests for the retrieval confidence gate."""

import pytest

from src.rag.retriever import confidence_gate


@pytest.mark.parametrize(
    "top,second,should_pass",
    [
        (0.80, 0.70, True),  # both thresholds met
        (0.65, 0.61, True),  # exactly at thresholds
        (0.64, 0.60, False),  # score just below threshold
        (0.50, 0.10, False),  # score too low
        (0.90, 0.00, True),  # single result, large gap
        # Small gap but both docs clear the absolute threshold: redundant
        # supporting chunks are corroboration, not ambiguity → pass.
        (0.80, 0.78, True),
        # Small gap and the runner-up is weak → genuinely ambiguous → fail.
        (0.66, 0.64, False),
    ],
)
def test_gate_absolute_scores(top: float, second: float, should_pass: bool) -> None:
    passed = confidence_gate(
        top, second, threshold=0.65, gap_threshold=0.03, absolute_scores=True
    )
    assert passed is should_pass


@pytest.mark.parametrize(
    "top,second,should_pass",
    [
        (1.00, 0.50, True),  # clear winner
        (1.00, 0.98, False),  # near-tie: relative scores get no waiver
        (0.40, 0.10, False),  # below threshold
    ],
)
def test_gate_relative_rrf_scores(top: float, second: float, should_pass: bool) -> None:
    passed = confidence_gate(
        top, second, threshold=0.5, gap_threshold=0.05, absolute_scores=False
    )
    assert passed is should_pass

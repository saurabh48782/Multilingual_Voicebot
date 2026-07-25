"""Manual RAGAS regression gate.

Opt-in via `uv run pytest -m eval`. Skipped by default because:
  - requires a running Ollama with the judge model (`evaluation.judge_model`) pulled
  - requires an ingested corpus (`data/index/`) and a generated testset
  - burns local GPU/compute (~hundreds of LLM calls per run)

Asserts each configured metric meets the threshold from
`params.yaml :: evaluation.thresholds`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.utils.config import cfg

pytestmark = pytest.mark.eval


def _testset_available() -> bool:
    return Path(cfg["evaluation"]["testset"]["path"]).is_file()


@pytest.mark.skipif(not _testset_available(), reason="testset.jsonl not generated yet")
def test_ragas_metrics_meet_thresholds(capfd: pytest.CaptureFixture[str]) -> None:
    from src.evaluation.ragas_eval import evaluate_testset

    eval_cfg = cfg["evaluation"]
    thresholds: dict[str, float] = eval_cfg["thresholds"]
    sample_size = int(os.environ.get("RAGAS_REGRESSION_SIZE", "10"))

    # This gate takes minutes-to-hours of local LLM calls. `evaluate_testset` already
    # emits a per-row tqdm bar plus RAGAS's own scoring bar, but pytest captures fd
    # 1/2 by default so the run looks hung. Release capture for the duration so the
    # bars reach the terminal without needing `-s`.
    with capfd.disabled():
        print(  # noqa: T201 - progress banner, capture is disabled here
            f"\n[ragas-gate] {sample_size} rows, languages=['en'] - "
            "phases: retrieve+generate, then RAGAS scoring",
            flush=True,
        )
        report = evaluate_testset(
            sample_size=sample_size,
            languages=["en"],
            output_dir=Path(eval_cfg["report_dir"]) / "regression",
        )

    en_metrics: dict[str, float | None] = report.get("results", {}).get("en", {}).get("metrics", {})
    failures: list[str] = []
    for name, threshold in thresholds.items():
        actual = en_metrics.get(name)
        if actual is None:
            failures.append(f"{name}: missing in report")
            continue
        if actual < float(threshold):
            failures.append(f"{name}: {actual:.3f} < {threshold:.3f}")

    assert not failures, "RAGAS regression failed:\n  " + "\n  ".join(failures)

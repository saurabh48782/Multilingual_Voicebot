"""CLI: `uv run python -m src.evaluation <subcommand>`.

Subcommands:
  generate  - synthesize a testset from the ingested corpus
  run       - execute RAGAS evaluation against an existing testset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.utils.config import cfg
from src.utils.logger import get_logger, setup_logging
from src.utils.observability import configure_tracing

logger = get_logger(__name__)


def _cmd_generate(args: argparse.Namespace) -> int:
    from src.evaluation.testset_gen import generate_testset

    languages = args.languages.split(",") if args.languages else None
    path = generate_testset(
        size=args.size,
        output_path=args.output,
        languages=languages,
    )
    print(f"testset → {path}")
    return 0


def _cmd_pretranslate(args: argparse.Namespace) -> int:
    from src.evaluation.pretranslate import pretranslate_testset

    languages = args.languages.split(",") if args.languages else None
    path = pretranslate_testset(
        testset_path=args.testset,
        languages=languages,
        model=args.translation_model,
        output_path=args.output,
    )
    print(f"translation cache → {path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from src.evaluation.ragas_eval import evaluate_testset

    metrics = args.metrics.split(",") if args.metrics else None
    languages = args.languages.split(",") if args.languages else None
    report = evaluate_testset(
        testset_path=args.testset,
        metrics=metrics,
        sample_size=args.sample_size,
        languages=languages,
        output_dir=args.output_dir,
        translation_model=args.translation_model,
        translation_cache=args.translation_cache,
    )
    summary = {
        lang: payload.get("metrics", {}) for lang, payload in (report.get("results") or {}).items()
    }
    print(json.dumps({"timestamp": report["timestamp"], "metrics": summary}, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    eval_cfg = cfg["evaluation"]
    p = argparse.ArgumentParser(prog="src.evaluation", description="RAGAS eval driver")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Synthesize testset from ingested corpus")
    g.add_argument("--size", type=int, default=eval_cfg["testset"]["size"])
    g.add_argument("--output", type=Path, default=None)
    g.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated subset of en,hi,bn (default: from params.yaml)",
    )
    g.set_defaults(func=_cmd_generate)

    r = sub.add_parser("run", help="Run RAGAS metrics on an existing testset")
    r.add_argument("--testset", type=Path, default=None)
    r.add_argument("--metrics", type=str, default=None, help="Comma-separated metric names")
    r.add_argument("--languages", type=str, default=None, help="Comma-separated languages to score")
    r.add_argument("--sample-size", type=int, default=None)
    r.add_argument("--output-dir", type=Path, default=None)
    r.add_argument(
        "--translation-model",
        type=str,
        default=None,
        help=(
            "Ollama model for the eval-time vernacular->English leg "
            "(default: evaluation.translation_model, else translation.ollama_model). "
            "Leaves the testset untouched."
        ),
    )
    r.add_argument(
        "--translation-cache",
        type=Path,
        default=None,
        help=(
            "JSONL from `pretranslate`; uses cached vernacular->English questions and "
            "loads no translation model (required when the translator does not fit on "
            "the GPU alongside the RAG + generation models)."
        ),
    )
    r.set_defaults(func=_cmd_run)

    t = sub.add_parser(
        "pretranslate",
        help="Cache the vernacular->English leg so a large translator runs in its own pass",
    )
    t.add_argument("--testset", type=Path, default=None)
    t.add_argument("--languages", type=str, default=None, help="Comma-separated vernaculars")
    t.add_argument("--translation-model", type=str, default=None)
    t.add_argument("--output", type=Path, default=None)
    t.set_defaults(func=_cmd_pretranslate)

    return p


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    configure_tracing()
    args = _build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())

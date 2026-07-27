"""Run the RAG pipeline over a testset and compute RAGAS metrics.

Mirrors the production retrieval + generation path so eval reflects real traffic.
See src/evaluation/README.md for the design decisions behind this module.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.evaluation.adapters import ragas_embeddings_wrapper, ragas_llm_wrapper
from src.evaluation.pretranslate import load_translation_cache
from src.graph.prompts import (
    GENERATE_PROMPT,
    GENERATE_SYSTEM,
    INSUFFICIENT_CONTEXT,
    sanitize_untrusted,
)
from src.llm.ollama_client import OllamaLLM
from src.rag.retriever import Retriever
from src.translation.base import SUPPORTED_VERNACULARS
from src.translation.ollama_translator import OllamaTranslator
from src.utils.config import ROOT_DIR, cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)


_METRIC_REGISTRY: dict[str, str] = {
    "context_precision": "ragas.metrics:LLMContextPrecisionWithReference",
    "context_recall": "ragas.metrics:LLMContextRecall",
    "faithfulness": "ragas.metrics:Faithfulness",
    "answer_relevancy": "ragas.metrics:ResponseRelevancy",
    "answer_correctness": "ragas.metrics:AnswerCorrectness",
    "answer_similarity": "ragas.metrics:SemanticSimilarity",
}

# Averaged over ANSWERED rows only; retrieval metrics stay over all rows. See README.
_ANSWER_DEPENDENT_METRICS: frozenset[str] = frozenset(
    {"faithfulness", "answer_relevancy", "answer_correctness", "answer_similarity"}
)


@dataclass
class EvalRow:
    question: str
    ground_truth: str
    contexts: list[str]
    answer: str
    language: str = "en"


def _instantiate_metric(name: str) -> Any:
    if name not in _METRIC_REGISTRY:
        raise ValueError(f"Unknown RAGAS metric: {name}")
    module_path, class_name = _METRIC_REGISTRY[name].split(":")
    mod = __import__(module_path, fromlist=[class_name])
    return getattr(mod, class_name)()


def _load_testset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"testset not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"testset at {path} is empty")
    return rows


def _generate_answer(llm: OllamaLLM, question_en: str, contexts: list[str]) -> str:
    """Mirror the production `generate` node so eval = real pipeline."""
    context = "\n\n---\n\n".join(
        f"[{i}] {sanitize_untrusted(c)}" for i, c in enumerate(contexts, 1)
    )
    resp = llm.complete(
        messages=[
            {"role": "system", "content": GENERATE_SYSTEM},
            {"role": "user", "content": GENERATE_PROMPT.format(context=context, query=question_en)},
        ],
        model=cfg["llm"]["model"],
        temperature=0.3,  # mirror the production generate node
        think=bool(cfg["llm"].get("think_on_generate", False)),
        stream=True,
    )
    answer = resp.content.strip()
    if answer.startswith(INSUFFICIENT_CONTEXT):
        return ""
    return answer  # type: ignore[no-any-return, unused-ignore]


def _run_pipeline_per_row(
    rows: list[dict[str, Any]],
    *,
    language: str,
    retriever: Retriever,
    llm: OllamaLLM,
    translator: OllamaTranslator | None,
    translation_cache: dict[tuple[str, str], str] | None = None,
) -> list[EvalRow]:
    """Execute retrieve+generate for each testset row in the requested language.

    Vernacular questions are translated to English before retrieval and the answer
    is kept in English, so RAGAS judges in English space. See README.
    """
    out: list[EvalRow] = []
    progress = tqdm(rows, desc=f"generate[{language}]", unit="row")
    for row in progress:
        if language == "en":
            question_vernac = row["question"]
            question_en = row["question"]
        else:
            translations = row.get("translations") or {}
            if language not in translations:
                logger.debug("row missing %s translation, skipping", language)
                continue
            question_vernac = translations[language]["question"]
            cached = (translation_cache or {}).get((language, question_vernac))
            if cached is not None:
                question_en = cached
            else:
                if translator is None:
                    if translation_cache is not None:
                        logger.warning(
                            "translation cache miss (%s) and no live translator - skipping row",
                            language,
                        )
                        continue
                    raise RuntimeError("translator required for vernacular eval")
                try:
                    question_en = translator.to_english(question_vernac, language).text
                except Exception:
                    logger.warning("row translation failed (%s) - skipping row", language)
                    continue

        result = retriever.search(question_en)
        contexts = [d.text_en for d in result.docs]
        # Empty answer == the three prod fallback paths: gate failed, model said
        # INSUFFICIENT_CONTEXT, or the LLM call errored. See README.
        if contexts and result.passed:
            try:
                answer = _generate_answer(llm, question_en, contexts)
            except Exception:
                logger.warning("row generation failed - degrading to empty answer")
                answer = ""
        else:
            answer = ""

        out.append(
            EvalRow(
                question=question_en,
                ground_truth=row["ground_truth"],
                contexts=contexts,
                answer=answer,
                language=language,
            )
        )
    return out


def _to_ragas_dataset(rows: list[EvalRow]) -> Any:
    """Build a RAGAS EvaluationDataset from EvalRow objects."""
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=r.question,
            retrieved_contexts=r.contexts,
            response=r.answer,
            reference=r.ground_truth,
        )
        for r in rows
    ]
    return EvaluationDataset(samples=samples)


def evaluate_testset(
    testset_path: str | Path | None = None,
    metrics: list[str] | None = None,
    sample_size: int | None = None,
    languages: list[str] | None = None,
    output_dir: str | Path | None = None,
    translation_model: str | None = None,
    translation_cache: str | Path | None = None,
) -> dict[str, Any]:
    """Run RAGAS evaluation and write a JSON+CSV report.

    Returns the in-memory report dict.
    """
    from ragas import evaluate
    from ragas.run_config import RunConfig

    eval_cfg = cfg["evaluation"]
    testset_path = Path(testset_path or eval_cfg["testset"]["path"])
    metrics = metrics or list(eval_cfg["metrics"])
    languages = languages or list(eval_cfg["testset"].get("languages", ["en"]))
    output_dir = Path(output_dir or eval_cfg["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_testset(testset_path)
    if sample_size and int(sample_size) < len(rows):
        n = int(sample_size)
        if eval_cfg.get("sample_strategy", "random") == "head":
            rows = rows[:n]
        else:
            rng = random.Random(eval_cfg.get("sample_seed", 42))  # noqa: S311 - test sampling, not crypto
            rows = rng.sample(rows, n)
    logger.info("Loaded %d testset rows from %s", len(rows), testset_path)

    # Decorrelation guard - see README ("Model choice").
    if eval_cfg["judge_model"] == cfg["llm"]["model"]:
        logger.warning(
            "RAGAS judge_model (%s) matches the generation model - metrics are "
            "self-graded and less defensible. Pin evaluation.judge_model to a "
            "different model family when GPU budget allows.",
            eval_cfg["judge_model"],
        )

    retriever = Retriever.load()
    llm = OllamaLLM()
    # Eval-time (vernacular -> English) translator only. The English -> vernacular
    # leg is baked into the testset at generate time, so overriding this ablates
    # one leg while leaving the testset byte-identical. See README.
    translation_model = (
        translation_model or eval_cfg.get("translation_model") or cfg["translation"]["ollama_model"]
    )
    # A cache (see pretranslate.py) lets a large translation model run in its own
    # pass, so it is never co-resident with the RAG + generation models on one GPU.
    cache = load_translation_cache(translation_cache) if translation_cache else None
    needs_translator = any(lng in SUPPORTED_VERNACULARS for lng in languages)
    translator = (
        OllamaTranslator(model=translation_model) if needs_translator and not cache else None
    )

    judge = ragas_llm_wrapper()
    embedder = ragas_embeddings_wrapper()
    metric_objs = [_instantiate_metric(m) for m in metrics]
    metric_columns = {name: obj.name for name, obj in zip(metrics, metric_objs, strict=True)}

    # Overrides RAGAS's timeout=180/max_workers=16 defaults, which NaN whole
    # metrics on a single serializing Ollama GPU. See README.
    rc_cfg = eval_cfg.get("run_config", {})
    run_config = RunConfig(
        timeout=int(rc_cfg.get("timeout", 300)),
        max_retries=int(rc_cfg.get("max_retries", 10)),
        max_wait=int(rc_cfg.get("max_wait", 60)),
        max_workers=int(rc_cfg.get("max_workers", 4)),
    )

    per_language: dict[str, Any] = {}
    started = time.time()

    for language in languages:
        logger.info("Generating answers for %s over %d rows (retrieve+LLM)", language, len(rows))
        eval_rows = _run_pipeline_per_row(
            rows,
            language=language,
            retriever=retriever,
            llm=llm,
            translator=translator,
            translation_cache=cache,
        )
        if not eval_rows:
            logger.warning("no evaluable rows for language=%s - skipping", language)
            continue
        dataset = _to_ragas_dataset(eval_rows)
        logger.info("Running RAGAS for %s on %d rows", language, len(eval_rows))
        result = evaluate(
            dataset=dataset,
            metrics=metric_objs,
            llm=judge,
            embeddings=embedder,
            run_config=run_config,
            show_progress=True,
        )
        scores_df = result.to_pandas()

        answered_mask = [bool(r.answer.strip()) for r in eval_rows]
        answered = sum(answered_mask)
        answered_rate = answered / len(eval_rows) if eval_rows else 0.0
        if len(scores_df) == len(answered_mask):
            answered_df = scores_df[answered_mask]
        else:
            logger.warning(
                "scores_df rows (%d) != eval rows (%d); reporting quality over all rows",
                len(scores_df),
                len(eval_rows),
            )
            answered_df = scores_df
        computed = {
            name: _safe_mean(
                answered_df if name in _ANSWER_DEPENDENT_METRICS else scores_df,
                metric_columns[name],
            )
            for name in metrics
        }
        computed["answered_rate"] = round(answered_rate, 4)
        per_language[language] = {
            "n_rows": len(eval_rows),
            "answered": answered,
            "metrics": computed,
            "rows": scores_df.to_dict(orient="records"),
        }

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "timestamp": timestamp,
        "duration_seconds": round(time.time() - started, 2),
        "testset_path": str(testset_path),
        "n_rows": len(rows),
        "languages": list(per_language.keys()),
        "metrics": metrics,
        "config_snapshot": {
            "judge_model": eval_cfg["judge_model"],
            "llm_model": cfg["llm"]["model"],
            "translation_model": translation_model,
            "translation_cache": str(translation_cache) if translation_cache else None,
            "embedding_model": eval_cfg["embedding_model"],
            "retrieval_threshold": cfg["rag"]["retrieval_threshold"],
            "retrieval_gap_threshold": cfg["rag"]["retrieval_gap_threshold"],
        },
        "results": per_language,
    }

    out_path = output_dir / f"report-{timestamp}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info("Report written to %s", out_path)
    return report


def _safe_mean(df: Any, column: str) -> float | None:
    if column not in df.columns:
        return None
    series = df[column].dropna()
    if series.empty:
        return None
    return float(series.mean())


def list_reports(report_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """List report metadata for the eval router."""
    report_dir = Path(report_dir or cfg["evaluation"]["report_dir"])
    if not report_dir.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for p in sorted(report_dir.glob("report-*.json"), reverse=True):
        try:
            data = json.loads(p.read_text())
            items.append(
                {
                    "id": p.stem,
                    "path": str(p.relative_to(ROOT_DIR)) if p.is_relative_to(ROOT_DIR) else str(p),
                    "timestamp": data.get("timestamp"),
                    "n_rows": data.get("n_rows"),
                    "languages": data.get("languages"),
                    "summary": {
                        lang: payload.get("metrics", {})
                        for lang, payload in (data.get("results") or {}).items()
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping unreadable report %s: %s", p, exc)
    return items


def load_report(report_id: str, report_dir: str | Path | None = None) -> dict[str, Any]:
    report_dir = Path(report_dir or cfg["evaluation"]["report_dir"])
    p = report_dir / f"{report_id}.json"
    if not p.is_file():
        raise FileNotFoundError(report_id)
    report: dict[str, Any] = json.loads(p.read_text())
    return report

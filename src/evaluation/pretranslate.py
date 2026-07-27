"""Pre-compute the eval-time vernacular -> English translation leg.

Exists because a large translation model and the RAG stack (e5-large +
bge-reranker) plus the generation model do not fit on one GPU simultaneously.
Running translation as a separate pass keeps only one heavy model resident at a
time, and the resulting cache makes re-runs of the same ablation free.

The testset is read-only here: this caches the hi/bn -> en leg only. The
en -> hi/bn leg is baked into the testset at generate time and is untouched,
so a cached run stays comparable to reports produced from the same testset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.translation.base import SUPPORTED_VERNACULARS
from src.translation.ollama_translator import OllamaTranslator
from src.utils.config import cfg
from src.utils.logger import get_logger

logger = get_logger(__name__)

CacheKey = tuple[str, str]


def _load_testset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"testset at {path} is empty")
    return rows


def load_translation_cache(path: str | Path) -> dict[CacheKey, str]:
    """Load a pretranslate JSONL into a {(language, vernacular_question): english} map."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"translation cache not found: {path}")
    cache: dict[CacheKey, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cache[(rec["language"], rec["question_vernac"])] = rec["question_en"]
    logger.info("Loaded %d cached translations from %s", len(cache), path)
    return cache


def pretranslate_testset(
    testset_path: str | Path | None = None,
    languages: list[str] | None = None,
    model: str | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Translate every vernacular testset question to English and cache the result.

    Resumable: an existing output file is loaded first and only missing
    (language, question) pairs are translated.
    """
    eval_cfg = cfg["evaluation"]
    testset_path = Path(testset_path or eval_cfg["testset"]["path"])
    model = model or eval_cfg.get("translation_model") or cfg["translation"]["ollama_model"]
    languages = [
        lng
        for lng in (languages or list(eval_cfg["testset"].get("languages", [])))
        if lng in SUPPORTED_VERNACULARS
    ]
    if not languages:
        raise ValueError("no vernacular languages to pretranslate")

    slug = str(model).replace(":", "-").replace("/", "-")
    out_path = Path(
        output_path or Path(eval_cfg["report_dir"]).parent / f"pretranslated-{slug}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[CacheKey, str] = {}
    if out_path.is_file():
        existing = load_translation_cache(out_path)

    rows = _load_testset(testset_path)
    pending: list[tuple[str, str]] = []
    for row in rows:
        translations = row.get("translations") or {}
        for lng in languages:
            if lng not in translations:
                continue
            question_vernac = translations[lng]["question"]
            if (lng, question_vernac) not in existing:
                pending.append((lng, question_vernac))

    logger.info(
        "Pretranslating %d pairs with %s (%d already cached)", len(pending), model, len(existing)
    )
    if not pending:
        return out_path

    translator = OllamaTranslator(model=model)
    failures = 0
    with out_path.open("a", encoding="utf-8") as f:
        for lng, question_vernac in tqdm(pending, desc=f"pretranslate[{model}]", unit="q"):
            try:
                question_en = translator.to_english(question_vernac, lng).text
            except Exception as exc:
                logger.warning("translation failed (%s): %s", lng, exc)
                failures += 1
                continue
            f.write(
                json.dumps(
                    {
                        "language": lng,
                        "question_vernac": question_vernac,
                        "question_en": question_en,
                        "model": model,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()

    logger.info("Wrote translation cache to %s (%d failures)", out_path, failures)
    return out_path

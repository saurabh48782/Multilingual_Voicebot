"""Synthesize a RAGAS testset from the already-ingested corpus.

English chunks from `metadata.parquet` → `ragas.testset.TestsetGenerator` → optional
translation into the vernaculars. See src/evaluation/README.md for the rationale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.documents import Document

from src.evaluation.adapters import ragas_embeddings_wrapper, ragas_llm_wrapper
from src.translation.base import SUPPORTED_VERNACULARS
from src.translation.ollama_translator import OllamaTranslator
from src.utils.config import cfg, faiss_metadata_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_corpus_documents(min_chars: int = 200) -> list[Document]:
    """Materialise ingested chunks as LangChain Documents (tiny chunks dropped)."""
    if not faiss_metadata_path.is_file():
        raise FileNotFoundError(
            f"No corpus metadata at {faiss_metadata_path}. "
            "Run ingestion first: uv run python -m src.data.ingestion.ingest"
        )
    df = pd.read_parquet(faiss_metadata_path)
    docs: list[Document] = []
    for _, row in df.iterrows():
        text = str(row.get("text_en", "")).strip()
        if len(text) < min_chars:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": row.get("doc_id"),
                    "chunk_id": row.get("chunk_id"),
                    "source": row.get("source"),
                    "page_num": int(row.get("page_num", 0)),
                },
            )
        )
    logger.info("Loaded %d corpus chunks for testset generation", len(docs))
    return docs


def _query_distribution() -> list[tuple[Any, float]]:
    """Translate params.yaml distribution dict into the RAGAS query-synth schema.

    Note: `simple` and `reasoning` both resolve to the single-hop synthesizer - RAGAS
    dropped distinct evolution types. See README ("query_distribution").
    """
    from ragas.testset.synthesizers import (
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    llm = ragas_llm_wrapper()
    cfg_dist = cfg["evaluation"]["testset"].get("query_distribution", {})

    mapping = {
        "simple": (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
        "reasoning": (SingleHopSpecificQuerySynthesizer(llm=llm), 0.25),
        "multi_context": (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),
    }
    out: list[tuple[Any, float]] = []
    for key, weight in cfg_dist.items():
        if key in mapping:
            synth, _ = mapping[key]
            out.append((synth, float(weight)))
    if not out:
        out = [(SingleHopSpecificQuerySynthesizer(llm=llm), 1.0)]
    return out


def generate_testset(
    size: int | None = None,
    output_path: str | Path | None = None,
    languages: list[str] | None = None,
) -> Path:
    """Generate a synthetic testset and persist it as JSONL.

    Each row:
        {
          "question": str,                # English
          "ground_truth": str,            # English reference answer
          "reference_contexts": list[str],
          "evolution_type": str,
          "translations": {               # only if vernaculars requested
              "hi": {"question": "...", "ground_truth": "..."},
              ...
          }
        }
    """
    from ragas.run_config import RunConfig
    from ragas.testset import TestsetGenerator

    eval_cfg = cfg["evaluation"]
    testset_cfg = eval_cfg["testset"]
    size = size or int(testset_cfg["size"])
    out = Path(output_path or testset_cfg["path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    langs = languages if languages is not None else list(testset_cfg.get("languages", ["en"]))

    docs = _load_corpus_documents()

    generator = TestsetGenerator(
        llm=ragas_llm_wrapper(),
        embedding_model=ragas_embeddings_wrapper(),
    )

    rc_cfg = testset_cfg.get("run_config", {})
    run_config = RunConfig(
        timeout=int(rc_cfg.get("timeout", 300)),
        max_retries=int(rc_cfg.get("max_retries", 10)),
        max_wait=int(rc_cfg.get("max_wait", 60)),
        max_workers=int(rc_cfg.get("max_workers", 4)),
    )
    raise_exceptions = bool(rc_cfg.get("raise_exceptions", False))

    logger.info("Generating %d synthetic test items", size)
    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=size,
        query_distribution=_query_distribution(),
        run_config=run_config,
        raise_exceptions=raise_exceptions,
    )
    df = testset.to_pandas()

    vernaculars = [lng for lng in langs if lng in SUPPORTED_VERNACULARS]
    translator = OllamaTranslator() if vernaculars else None

    with out.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            question = str(row.get("user_input") or row.get("question") or "")
            ground_truth = str(row.get("reference") or row.get("ground_truth") or "")
            contexts_raw = row.get("reference_contexts") or row.get("contexts") or []
            contexts = list(contexts_raw) if hasattr(contexts_raw, "__iter__") else []
            item: dict[str, Any] = {
                "question": question,
                "ground_truth": ground_truth,
                "reference_contexts": contexts,
                "evolution_type": str(
                    row.get("synthesizer_name") or row.get("evolution_type") or ""
                ),
            }
            if translator and vernaculars:
                translations: dict[str, dict[str, str]] = {}
                for lng in vernaculars:
                    try:
                        q_t = translator.to_vernacular(question, lng).text
                        a_t = translator.to_vernacular(ground_truth, lng).text
                        translations[lng] = {"question": q_t, "ground_truth": a_t}
                    except Exception as exc:  # noqa: BLE001 - log + skip
                        logger.warning("translation failed for %s: %s", lng, exc)
                if translations:
                    item["translations"] = translations
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Testset written to %s (%d rows)", out, len(df))
    return out

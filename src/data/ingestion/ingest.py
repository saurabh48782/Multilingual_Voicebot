"""CLI for corpus ingestion.

Usage:
    # Ingest entire data/corpus/ directory:
    uv run python -m src.data.ingestion.ingest

    # Ingest a single file:
    uv run python -m src.data.ingestion.ingest --file data/corpus/filename.pdf

    # Re-ingest everything (ignore manifest):
    uv run python -m src.data.ingestion.ingest --force

    # Skip translation (corpus already in English):
    uv run python -m src.data.ingestion.ingest --no-translate

    # Show index stats:
    uv run python -m src.data.ingestion.ingest --stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("data.ingestion")


def get_stats() -> None:
    from src.rag.store import get_store

    store = get_store()
    logger.info(
        "Index stats", total_chunks=store.total_chunks, index_path=str(store.index_path)
    )
    print(f"Index chunks : {store.total_chunks}")
    print(f"Index path   : {store.index_path}")
    print(f"Metadata path: {store.metadata_path}")
    if len(store._meta):
        docs = store._meta["doc_id"].unique().tolist()
        print(f"Total Documents    : {len(docs)}")
        for doc in sorted(docs):
            n = len(store._meta[store._meta["doc_id"] == doc])
            print(f"{doc} has : {n} chunks")


def ingest(args: argparse.Namespace) -> None:
    translate = not args.no_translate

    if args.file:
        path = Path(args.file)
        if not path.exists():
            logger.error("File not found", path=str(path))
            sys.exit(1)
        from src.rag.ingestor import ingest_file

        count = ingest_file(path, force=args.force, translate=translate)
        logger.info("Ingestion complete", file=path.name, chunks_added=count)
        print(f"Done. {count} chunks added from {path.name}.")
    else:
        from src.rag.ingestor import ingest_corpus
        from src.utils.config import corpus_dir

        logger.info(
            "Ingesting corpus",
            corpus_dir=str(corpus_dir),
            force=args.force,
            translate=translate,
        )
        summary = ingest_corpus(corpus_dir, force=args.force, translate=translate)
        total = sum(summary.values())
        logger.info("Corpus ingestion complete", total_chunks=total, files=len(summary))
        print(f"\nDone. {total} chunks added across {len(summary)} file(s).")
        for fname, n in sorted(summary.items()):
            print(f" {fname}: {n} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Voicebot corpus ingestion")
    parser.add_argument("--file", metavar="PATH", help="Ingest a single file")
    parser.add_argument(
        "--force", action="store_true", help="Re-ingest even if unchanged"
    )
    parser.add_argument(
        "--no-translate", action="store_true", help="Skip Groq translation"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show index stats and exit"
    )
    args = parser.parse_args()

    if args.stats:
        get_stats()
    else:
        ingest(args)


if __name__ == "__main__":
    main()

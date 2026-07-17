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

    # Clear all FAISS + BM25 artifacts and the manifest (wipe the index):
    uv run python -m src.data.ingestion.ingest --clear
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("data.ingestion")


def get_stats() -> None:
    from src.rag.ingestor import index_stats
    from src.rag.store import get_store

    store = get_store()
    stats = index_stats()
    logger.info("Index stats", total_chunks=stats["total_chunks"], index_path=str(store.index_path))
    print(f"Index chunks : {stats['total_chunks']}")
    print(f"Index path   : {store.index_path}")
    print(f"Metadata path: {store.metadata_path}")
    if stats["documents"]:
        print(f"Total Documents    : {stats['total_documents']}")
        for doc in stats["documents"]:
            print(f"{doc['doc_id']} has : {doc['chunks']} chunks")


def clear() -> None:
    from src.rag.ingestor import clear_index

    removed = clear_index()
    if removed:
        print(f"Cleared {len(removed)} index artifact(s):")
        for path in removed:
            print(f"  removed {path}")
    else:
        print("Nothing to clear; index is already empty.")


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
    parser.add_argument("--force", action="store_true", help="Re-ingest even if unchanged")
    parser.add_argument("--no-translate", action="store_true", help="Skip translation")
    parser.add_argument("--stats", action="store_true", help="Show index stats and exit")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all FAISS + BM25 artifacts and the manifest, then exit",
    )
    args = parser.parse_args()

    if args.clear:
        clear()
    elif args.stats:
        get_stats()
    else:
        ingest(args)


if __name__ == "__main__":
    main()

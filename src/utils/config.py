import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

# Derived paths used across multiple modules
faiss_index_path = DATA_DIR / "index" / "faiss.index"
faiss_metadata_path = DATA_DIR / "index" / "metadata.parquet"
faiss_manifest_path = DATA_DIR / "index" / "manifest.json"
bm25_index_dir = DATA_DIR / "index" / "bm25"
bm25_corpus_path = DATA_DIR / "index" / "bm25_corpus.pkl"
corpus_dir = DATA_DIR / "corpus"
audio_cache_dir = DATA_DIR / "audio_cache"


def _replace_env_vars(config_str: str) -> str:
    """Replace ${ENV_VAR} or ${ENV_VAR:-default} placeholders with environment values.

    ${VAR}        - raises RuntimeError when unset (required)
    ${VAR:-}      - returns empty string when unset (optional)
    ${VAR:-value} - returns 'value' when unset (optional with default)
    """
    pattern = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")

    def _lookup(match: re.Match[str]) -> str:
        name = match.group(1)
        default: str | None = match.group(2)  # None when no :- present
        value = os.environ.get(name)
        if value is not None:
            return value
        if default is not None:
            return default
        raise RuntimeError(f"Environment variable '${{{name}}}' referenced in config but not set.")

    return pattern.sub(_lookup, config_str)


@lru_cache(maxsize=1)
def load_config(config_path: str = "params.yaml") -> dict[str, Any]:
    """Load YAML config and resolve ${ENV_VAR} placeholders.

    Cached for process lifetime - restart required to pick up file or env changes.
    """
    path = Path(config_path)
    if not path.is_file():
        logger.error("Configuration file not found: %s", path)
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    logger.debug("Loading configuration from %s", path)
    with open(path, encoding="utf-8") as f:
        config_str = f.read()

    resolved = _replace_env_vars(config_str)
    config: dict[str, Any] = yaml.safe_load(resolved)

    logger.debug("Configuration loaded, keys: %s", list(config.keys()) if config else [])
    return config


def validate_config() -> None:
    """Validate critical config at startup so problems surface immediately."""
    cfg = load_config(str(ROOT_DIR / "params.yaml"))

    # Required top-level sections — a trimmed params.yaml (missing memory/
    # evaluation) otherwise fails late with an opaque KeyError deep in db.py
    # startup or at pytest collection. Surface it here instead.
    for section in ("llm", "rag", "stt", "tts", "memory", "evaluation"):
        if section not in cfg:
            raise RuntimeError(f"Missing required config section '{section}' in params.yaml.")

    rag = cfg.get("rag", {})
    for name in ("retrieval_threshold", "retrieval_gap_threshold"):
        value = rag.get(name)
        if value is not None and (not isinstance(value, int | float) or not (0.0 < value <= 1.0)):
            raise RuntimeError(f"Invalid rag.{name}: {value!r} - must be a number in (0, 1].")


cfg = load_config(str(ROOT_DIR / "params.yaml"))

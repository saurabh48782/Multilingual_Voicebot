import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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

    ${VAR}        - raises RuntimeError when unset or empty (required)
    ${VAR:-}      - returns empty string when unset or empty (optional)
    ${VAR:-value} - returns 'value' when unset or empty (optional with default)

    Matches shell ${VAR:-default} semantics: a var that is *set but empty* is
    treated the same as unset, so the default fires. (Needed because direnv/.env
    export bare-metal-only vars like STT_REMOTE_URL as empty strings on the host.)
    """
    pattern = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")

    def _lookup(match: re.Match[str]) -> str:
        name = match.group(1)
        default: str | None = match.group(2)  # None when no :- present
        value = os.environ.get(name)
        if value:  # set and non-empty (shell :- treats "" like unset)
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
    """Validate critical config at startup."""
    cfg = load_config(str(ROOT_DIR / "params.yaml"))

    for section in ("llm", "rag", "stt", "tts", "memory", "evaluation", "db"):
        if section not in cfg:
            raise RuntimeError(f"Missing required config section '{section}' in params.yaml.")

    rag = cfg.get("rag", {})
    for name in ("retrieval_threshold", "retrieval_gap_threshold"):
        value = rag.get(name)
        if value is not None and (not isinstance(value, int | float) or not (0.0 < value <= 1.0)):
            raise RuntimeError(f"Invalid rag.{name}: {value!r} - must be a number in (0, 1].")

    _validate_db_dsn(cfg.get("memory", {}).get("checkpoint_dsn", ""))


def _validate_db_dsn(dsn: str) -> None:
    """Check the DB DSN the app actually connects with (memory.checkpoint_dsn) is
    a well-formed postgresql:// URL with host/user/password/dbname all present.

    Only the resolved DSN is checked - not the individual db.* pieces in
    params.yaml, since those are documentation/bare-metal defaults and are not
    guaranteed to be populated inside every deployment (e.g. the docker-compose
    `api` container only ever sets CHECKPOINT_DSN, not POSTGRES_PASSWORD)."""
    if not dsn:
        raise RuntimeError("memory.checkpoint_dsn is not set - export CHECKPOINT_DSN (see .env).")
    parsed = urlsplit(dsn)
    if parsed.scheme not in ("postgresql", "postgres"):
        raise RuntimeError(
            f"memory.checkpoint_dsn has invalid scheme {parsed.scheme!r} - expected postgresql://."
        )
    if not parsed.hostname:
        raise RuntimeError("memory.checkpoint_dsn is missing a host.")
    if not parsed.username:
        raise RuntimeError("memory.checkpoint_dsn is missing a username.")
    if not parsed.password:
        raise RuntimeError("memory.checkpoint_dsn is missing a password.")
    if not parsed.path.lstrip("/"):
        raise RuntimeError("memory.checkpoint_dsn is missing a database name.")


cfg = load_config(str(ROOT_DIR / "params.yaml"))

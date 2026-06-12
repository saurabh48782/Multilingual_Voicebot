"""
Centralized structured logging configuration using structlog.
Usage:

# Local development (file logs → ``jq``)
    jq 'select(.session_id == "abc123")' logs/log_2026-06-12.log

# Container / stdout (LOG_TARGET=stdout → one JSON object per line)
    docker compose logs api | jq 'select(.session_id == "abc123")'
"""

import logging
import logging.handlers
import os
import sys
from contextvars import ContextVar
from datetime import datetime

import structlog

# Tracks the session id for the current async task / request so it can be
# bound onto every log event via ``structlog.contextvars.bind_contextvars``.
current_session: ContextVar[str | None] = ContextVar("current_session", default=None)

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")

_initialized = False


def _today_log_file() -> str:
    """Compute the log file path for today's date (call-time, not import-time)."""
    return os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")


# Shared processors applied to every log event (both structlog-native
# loggers and stdlib loggers captured via ProcessorFormatter).
_shared_processors: list[structlog.types.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.stdlib.ExtraAdder(),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.UnicodeDecoder(),
]

# EARLY structlog configuration
structlog.configure(
    processors=[
        *_shared_processors,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
) -> None:
    """
    Set up structured logging with console output and optional JSON file output.

    When ``LOG_TARGET=stdout`` (recommended for containerized / Fargate
    deployments where CloudWatch is the log sink), the file handler is
    skipped entirely so no ephemeral disk I/O is wasted.
    """
    global _initialized
    if _initialized:
        return

    is_tty = sys.stdout.isatty()
    log_target = os.environ.get("LOG_TARGET", "").lower()
    is_cloud = log_target == "stdout"
    use_colors = is_tty and not is_cloud
    enable_file_logging = log_target != "stdout"

    if enable_file_logging:
        if log_file is None:
            log_file = _today_log_file()
        os.makedirs(LOGS_DIR, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    # Console formatter: human-readable, colored
    if use_colors:
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=_shared_processors,
    )

    # Attach handlers to the root stdlib logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(console_formatter)
    root.addHandler(console)

    if enable_file_logging:
        assert log_file is not None  # nosec B101 - type narrowing; guarded by enable_file_logging
        # File formatter: JSON (one object per line)
        json_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=_shared_processors,
        )
        file_handler = logging.handlers.WatchedFileHandler(
            log_file,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(json_formatter)
        root.addHandler(file_handler)

    # Quieten noisy third-party loggers
    for name in (
        "httpx",
        "httpcore",
        "asyncio",
        "urllib3",
        "uvicorn.access",
        "transformers",
        "sentence_transformers",
        "faiss",
        "datasets",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    _initialized = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger for a specific module."""
    return structlog.get_logger(f"voicebot.{name}")

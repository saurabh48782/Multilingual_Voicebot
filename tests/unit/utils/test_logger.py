"""Unit tests for src/utils/logger.py."""

import logging
import logging.handlers
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

import src.utils.logger
from src.utils.logger import (
    LOGS_DIR,
    _today_log_file,
    current_session,
    get_logger,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_logging_state() -> Iterator[None]:
    """
    Cleanly reset Python's global logging state and the module's
    ``_initialized`` flag around every test.
    """
    original_initialized = getattr(src.utils.logger, "_initialized", False)
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    # Reset module initialization flag so each test exercises a fresh setup.
    src.utils.logger._initialized = False

    yield

    root_logger.handlers = original_handlers
    root_logger.setLevel(original_level)
    src.utils.logger._initialized = original_initialized


@pytest.fixture(autouse=True)
def clear_log_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to file logging (LOG_TARGET unset) unless a test opts out."""
    monkeypatch.delenv("LOG_TARGET", raising=False)


class TestGetLogger:
    @pytest.mark.parametrize("module_name", ["test_module", "api", "rag.retriever"])
    def test_returns_prefixed_bound_logger(self, module_name: str) -> None:
        logger = get_logger(module_name)
        assert logger is not None
        assert logger.name == f"voicebot.{module_name}"


class TestCurrentSession:
    def test_default_is_none(self) -> None:
        assert current_session.get() is None

    @pytest.mark.parametrize("session_id", ["abc123", "session-1", ""])
    def test_set_and_get(self, session_id: str) -> None:
        token = current_session.set(session_id)
        try:
            assert current_session.get() == session_id
        finally:
            current_session.reset(token)


class TestTodayLogFile:
    def test_returns_path_in_logs_dir(self) -> None:
        path = _today_log_file()
        assert path.startswith(LOGS_DIR)
        assert "log_" in path
        assert path.endswith(".log")


class TestSetupLogging:
    @pytest.mark.parametrize(
        "log_level, expected_level",
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("UNKNOWN", logging.INFO),  # Fallback when string is not a valid level
        ],
    )
    def test_setup_logging_levels_and_handlers(
        self, tmp_path: Path, log_level: str, expected_level: int
    ) -> None:
        mock_logs_dir = tmp_path / "my_logs"

        log_file = str(tmp_path / "test.log")

        with patch("src.utils.logger.LOGS_DIR", str(mock_logs_dir)):
            setup_logging(log_level=log_level, log_file=log_file)

        # Directory creation
        assert mock_logs_dir.exists()

        # Root logger level
        root = logging.getLogger()
        assert root.level == expected_level

        # Console (stream) + file handlers attached
        assert len(root.handlers) == 2
        handler_types = [type(h) for h in root.handlers]
        assert logging.StreamHandler in handler_types
        assert logging.handlers.WatchedFileHandler in handler_types

    def test_setup_logging_stdout_target_skips_file_handler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LOG_TARGET=stdout skips file logging - only the console handler."""
        monkeypatch.setenv("LOG_TARGET", "stdout")

        with patch("src.utils.logger.LOGS_DIR", str(tmp_path)):
            setup_logging(log_level="INFO")

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
        assert not any(isinstance(h, logging.handlers.WatchedFileHandler) for h in root.handlers)

    @pytest.mark.parametrize(
        "noisy_logger_name",
        [
            "httpx",
            "httpcore",
            "asyncio",
            "urllib3",
            "uvicorn.access",
            "transformers",
            "sentence_transformers",
            "faiss",
            "datasets",
        ],
    )
    def test_setup_logging_quiets_noisy_loggers(
        self, tmp_path: Path, noisy_logger_name: str
    ) -> None:
        """Chatty third-party libraries are pinned to WARNING."""
        log_file = str(tmp_path / "test.log")

        with patch("src.utils.logger.LOGS_DIR", str(tmp_path)):
            setup_logging(log_level="DEBUG", log_file=log_file)

        assert logging.getLogger(noisy_logger_name).level == logging.WARNING

    def test_setup_logging_is_idempotent(self, tmp_path: Path) -> None:
        """Second call returns early without duplicating handlers."""
        log_file = str(tmp_path / "test.log")

        with patch("src.utils.logger.LOGS_DIR", str(tmp_path)):
            setup_logging(log_level="DEBUG", log_file=log_file)
            root = logging.getLogger()
            handlers_count_after_first = len(root.handlers)

            setup_logging(log_level="DEBUG", log_file=log_file)

            assert len(root.handlers) == handlers_count_after_first

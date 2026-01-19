"""Tests for configure_logging handler setup and formatting."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from unittest.mock import patch

import pytest

from gpt_trader.logging.json_formatter import StructuredJSONFormatterWithTimestamp
from gpt_trader.logging.setup import configure_logging


class TestConfigureLoggingHandlers:
    """Test handler creation and formatter behavior in configure_logging."""

    @pytest.fixture(autouse=True)
    def clean_logging_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in (
            "COINBASE_TRADER_LOG_DIR",
            "PERPS_LOG_DIR",
            "COINBASE_TRADER_LOG_MAX_BYTES",
            "PERPS_LOG_MAX_BYTES",
            "COINBASE_TRADER_LOG_BACKUP_COUNT",
            "PERPS_LOG_BACKUP_COUNT",
            "COINBASE_TRADER_CRIT_LOG_MAX_BYTES",
            "PERPS_CRIT_LOG_MAX_BYTES",
            "COINBASE_TRADER_CRIT_LOG_BACKUP_COUNT",
            "PERPS_CRIT_LOG_BACKUP_COUNT",
            "COINBASE_TRADER_DEBUG",
            "PERPS_DEBUG",
        ):
            monkeypatch.delenv(key, raising=False)

    @pytest.fixture(autouse=True)
    def reset_logging(self) -> None:
        """Reset logging configuration before each test."""
        # Clear all handlers from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

        # Clear all handlers from json logger
        json_logger = logging.getLogger("gpt_trader.json")
        for handler in json_logger.handlers[:]:
            json_logger.removeHandler(handler)
            handler.close()

        # Reset log levels
        root_logger.setLevel(logging.WARNING)
        json_logger.setLevel(logging.WARNING)

        yield

        # Cleanup after test
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        for handler in json_logger.handlers[:]:
            json_logger.removeHandler(handler)
            handler.close()

    def test_configure_logging_creates_directories(self, tmp_path: Path) -> None:
        """Test that logging directories are created in the configured log dir."""
        log_dir = tmp_path / "logs"
        with patch("gpt_trader.logging.setup.LOG_DIR", log_dir):
            configure_logging()

        assert log_dir.exists()

    def test_configure_logging_creates_console_handler(self, tmp_path: Path) -> None:
        """Test that console handler is created."""
        with patch("gpt_trader.logging.setup.LOG_DIR", tmp_path / "logs"):
            configure_logging()

        root_logger = logging.getLogger()
        # Check that we have stream handlers (console or otherwise)
        stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1
        # At least one should be configured for INFO level (root logger is INFO)
        assert root_logger.level == logging.INFO

    def test_configure_logging_creates_file_handlers(self, tmp_path: Path) -> None:
        """Test that file handlers are created."""
        with patch("gpt_trader.logging.setup.LOG_DIR", tmp_path / "logs"):
            configure_logging()

        root_logger = logging.getLogger()
        file_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        # Should have at least 2 file handlers (general + critical)
        assert len(file_handlers) >= 2

    def test_configure_logging_creates_json_logger(self, tmp_path: Path) -> None:
        """Test that JSON logger is configured."""
        with patch("gpt_trader.logging.setup.LOG_DIR", tmp_path / "logs"):
            configure_logging()

        json_logger = logging.getLogger("gpt_trader.json")
        assert json_logger.level == logging.DEBUG
        assert json_logger.propagate is False
        assert len(json_logger.handlers) >= 2  # general + critical

    def test_configure_logging_json_handlers(self, tmp_path: Path) -> None:
        """Test that JSON logger handlers are properly configured."""
        with patch("gpt_trader.logging.setup.LOG_DIR", tmp_path / "logs"):
            configure_logging()

        json_logger = logging.getLogger("gpt_trader.json")
        handlers = json_logger.handlers

        # Check we have rotating file handlers
        rotating_handlers = [
            h for h in handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(rotating_handlers) >= 2

        # Check that handlers have the right formatters (structured JSON)
        for handler in rotating_handlers:
            assert isinstance(handler.formatter, StructuredJSONFormatterWithTimestamp)

    def test_configure_logging_critical_handler_level(self, tmp_path: Path) -> None:
        """Test that critical handler is set to WARNING level."""
        with patch("gpt_trader.logging.setup.LOG_DIR", tmp_path / "logs"):
            configure_logging()

        root_logger = logging.getLogger()
        rotating_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]

        # Should have at least one handler at WARNING level (critical handler)
        warning_handlers = [h for h in rotating_handlers if h.level == logging.WARNING]
        assert len(warning_handlers) >= 1

    def test_configure_logging_handler_formatters(self, tmp_path: Path) -> None:
        """Test that handlers have formatters configured."""
        with patch("gpt_trader.logging.setup.LOG_DIR", tmp_path / "logs"):
            configure_logging()

        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            assert handler.formatter is not None
            # Check that formatter has the expected fields
            fmt = handler.formatter._fmt
            assert "asctime" in fmt or "message" in fmt
            assert "levelname" in fmt or "message" in fmt

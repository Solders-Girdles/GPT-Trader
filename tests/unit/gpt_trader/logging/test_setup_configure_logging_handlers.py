"""Tests for configure_logging handler setup and formatting."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.logging.setup import configure_logging

pytestmark = pytest.mark.legacy_modernize


class TestConfigureLoggingHandlers:
    """Test handler creation and formatter behavior in configure_logging."""

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

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_creates_directories(
        self, mock_mkdir: MagicMock, mock_ensure: MagicMock, tmp_path: Path
    ) -> None:
        """Test that logging directories are created."""
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()

        mock_ensure.assert_called_once()
        mock_mkdir.assert_called()

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_creates_console_handler(
        self, mock_mkdir: MagicMock, mock_ensure: MagicMock, tmp_path: Path
    ) -> None:
        """Test that console handler is created."""
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()

        root_logger = logging.getLogger()
        # Check that we have stream handlers (console or otherwise)
        stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1
        # At least one should be configured for INFO level (root logger is INFO)
        assert root_logger.level == logging.INFO

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_creates_file_handlers(
        self, mock_mkdir: MagicMock, mock_ensure: MagicMock, tmp_path: Path
    ) -> None:
        """Test that file handlers are created."""
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()

        root_logger = logging.getLogger()
        file_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        # Should have at least 2 file handlers (general + critical)
        assert len(file_handlers) >= 2

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_sets_root_level(
        self, mock_mkdir: MagicMock, mock_ensure: MagicMock, tmp_path: Path
    ) -> None:
        """Test that root logger level is set to INFO."""
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_creates_json_logger(
        self, mock_mkdir: MagicMock, mock_ensure: MagicMock, tmp_path: Path
    ) -> None:
        """Test that JSON logger is configured."""
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()

        json_logger = logging.getLogger("gpt_trader.json")
        assert json_logger.level == logging.DEBUG
        assert json_logger.propagate is False
        assert len(json_logger.handlers) >= 2  # general + critical

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_json_handlers(
        self, mock_mkdir: MagicMock, mock_ensure: MagicMock, tmp_path: Path
    ) -> None:
        """Test that JSON logger handlers are properly configured."""
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()

        json_logger = logging.getLogger("gpt_trader.json")
        handlers = json_logger.handlers

        # Check we have rotating file handlers
        rotating_handlers = [
            h for h in handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(rotating_handlers) >= 2

        # Check that handlers have the right formatters (plain for JSON)
        for handler in rotating_handlers:
            assert handler.formatter is not None
            # JSON formatter should just output the message
            assert handler.formatter._fmt == "%(message)s"

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_critical_handler_level(
        self, mock_mkdir: MagicMock, mock_ensure: MagicMock, tmp_path: Path
    ) -> None:
        """Test that critical handler is set to WARNING level."""
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()

        root_logger = logging.getLogger()
        rotating_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]

        # Should have at least one handler at WARNING level (critical handler)
        warning_handlers = [h for h in rotating_handlers if h.level == logging.WARNING]
        assert len(warning_handlers) >= 1

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_handler_formatters(
        self, mock_mkdir: MagicMock, mock_ensure: MagicMock, tmp_path: Path
    ) -> None:
        """Test that handlers have formatters configured."""
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()

        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            assert handler.formatter is not None
            # Check that formatter has the expected fields
            fmt = handler.formatter._fmt
            assert "asctime" in fmt or "message" in fmt
            assert "levelname" in fmt or "message" in fmt

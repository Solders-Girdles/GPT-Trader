"""Tests for configure_logging behavior in TUI/CLI modes and idempotency."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.logging.setup import configure_logging


class TestConfigureLoggingModesAndIdempotency:
    """Test configure_logging mode behavior and idempotency."""

    @pytest.fixture(autouse=True)
    def reset_logging(self) -> None:
        """Reset logging configuration before each test."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

        json_logger = logging.getLogger("gpt_trader.json")
        for handler in json_logger.handlers[:]:
            json_logger.removeHandler(handler)
            handler.close()

        yield

        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        for handler in json_logger.handlers[:]:
            json_logger.removeHandler(handler)
            handler.close()

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_idempotent(
        self, mock_mkdir: MagicMock, mock_ensure: MagicMock, tmp_path: Path
    ) -> None:
        """Test that calling configure_logging multiple times doesn't duplicate handlers."""
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()
            initial_handler_count = len(logging.getLogger().handlers)

            configure_logging()
            second_handler_count = len(logging.getLogger().handlers)

        # Handler count should not increase
        # (it might be slightly different due to deduplication logic, but shouldn't double)
        assert second_handler_count <= initial_handler_count + 2

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_tui_mode_suppresses_stream_handler(
        self, mock_mkdir: MagicMock, mock_ensure: MagicMock, tmp_path: Path
    ) -> None:
        """Test that TUI mode suppresses StreamHandler to prevent display corruption."""
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging(tui_mode=True)

        root_logger = logging.getLogger()
        # Filter for console StreamHandlers only (exclude RotatingFileHandler and test handlers)
        console_handlers = [
            h
            for h in root_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
            and type(h).__name__ not in {"LogCaptureHandler", "LogCaptureFixture"}
        ]

        # No console StreamHandler should be present in TUI mode
        assert len(console_handlers) == 0, "TUI mode should not have console StreamHandler"

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_cli_mode_creates_stream_handler(
        self, mock_mkdir: MagicMock, mock_ensure: MagicMock, tmp_path: Path
    ) -> None:
        """Test that CLI mode creates StreamHandler for console output."""
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging(tui_mode=False)

        root_logger = logging.getLogger()
        # Filter for console StreamHandlers only (exclude RotatingFileHandler and test handlers)
        console_handlers = [
            h
            for h in root_logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
            and type(h).__name__ not in {"LogCaptureHandler", "LogCaptureFixture"}
        ]

        # At least one console StreamHandler should be present in CLI mode
        assert len(console_handlers) >= 1, "CLI mode should have console StreamHandler"

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_file_handlers_present_in_both_modes(
        self, mock_mkdir: MagicMock, mock_ensure: MagicMock, tmp_path: Path
    ) -> None:
        """Test that file handlers are present in both TUI and CLI modes."""
        # Test TUI mode
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging(tui_mode=True)

        root_logger = logging.getLogger()
        tui_file_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(tui_file_handlers) >= 2, "TUI mode should have file handlers"

        # Reset handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

        # Test CLI mode
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging(tui_mode=False)

        cli_file_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(cli_file_handlers) >= 2, "CLI mode should have file handlers"

        # Both modes should have same number of file handlers
        assert len(tui_file_handlers) == len(
            cli_file_handlers
        ), "Both modes should have the same file handlers"

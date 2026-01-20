"""Smoke tests for logging setup."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.logging.setup import configure_logging


class TestLoggingSetupSmoke:
    """Smoke tests for logging functionality."""

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
    def test_logging_works_after_configuration(
        self, mock_mkdir: MagicMock, mock_ensure: MagicMock, tmp_path: Path
    ) -> None:
        """Test that logging actually works after configuration."""
        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()

        # Should not raise any exceptions
        logger = logging.getLogger("test_logger")
        logger.info("Test message")
        logger.warning("Test warning")
        logger.error("Test error")
        root_logger = logging.getLogger()
        file_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) >= 2
        assert logging.getLogger("gpt_trader.json").handlers

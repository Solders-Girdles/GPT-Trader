"""Tests for configure_logging behavior under environment overrides."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.logging.setup import configure_logging


class TestConfigureLoggingEnvOverrides:
    """Test configure_logging env var overrides."""

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
    def test_configure_logging_respects_env_log_dir(
        self, mock_ensure: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that PERPS_LOG_DIR environment variable is respected."""
        custom_log_dir = tmp_path / "custom_logs"
        custom_log_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PERPS_LOG_DIR", str(custom_log_dir))

        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()

        # Check that file handlers were created in the custom directory
        root_logger = logging.getLogger()
        rotating_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        # At least one handler should use the custom log directory
        handler_paths = [getattr(h, "baseFilename", "") for h in rotating_handlers]
        assert any(str(custom_log_dir) in path for path in handler_paths)

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_respects_max_bytes_env(
        self,
        mock_mkdir: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that log size environment variables are respected."""
        custom_max_bytes = 1024 * 1024  # 1 MB
        monkeypatch.setenv("PERPS_LOG_MAX_BYTES", str(custom_max_bytes))

        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()

        root_logger = logging.getLogger()
        rotating_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]

        # At least one handler should have the custom max bytes
        assert any(h.maxBytes == custom_max_bytes for h in rotating_handlers)

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_respects_backup_count_env(
        self,
        mock_mkdir: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that backup count environment variables are respected."""
        custom_backup_count = 20
        monkeypatch.setenv("PERPS_LOG_BACKUP_COUNT", str(custom_backup_count))

        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()

        root_logger = logging.getLogger()
        rotating_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]

        # At least one handler should have the custom backup count
        assert any(h.backupCount == custom_backup_count for h in rotating_handlers)

    @patch("gpt_trader.logging.setup.ensure_directories")
    @patch("pathlib.Path.mkdir")
    def test_configure_logging_enables_debug_with_env(
        self,
        mock_mkdir: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that PERPS_DEBUG environment variable enables debug logging."""
        monkeypatch.setenv("PERPS_DEBUG", "1")

        with patch("gpt_trader.logging.setup.LOG_DIR", str(tmp_path)):
            configure_logging()

        # Check that specific loggers are set to DEBUG
        coinbase_logger = logging.getLogger("gpt_trader.features.brokerages.coinbase")
        live_trade_logger = logging.getLogger("gpt_trader.features.live_trade")

        assert coinbase_logger.level == logging.DEBUG
        assert live_trade_logger.level == logging.DEBUG

"""Tests for configure_logging behavior under environment overrides."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import pytest

import gpt_trader.logging.setup as logging_setup
from gpt_trader.logging.setup import configure_logging


def _clear_handlers() -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    json_logger = logging.getLogger("gpt_trader.json")
    for handler in json_logger.handlers[:]:
        json_logger.removeHandler(handler)
        handler.close()


def _rotating_handlers(logger: logging.Logger) -> list[logging.handlers.RotatingFileHandler]:
    return [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.handlers.RotatingFileHandler)
    ]


@pytest.fixture(autouse=True)
def log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in (
        "COINBASE_TRADER_LOG_DIR",
        "PERPS_LOG_DIR",
        "COINBASE_TRADER_LOG_MAX_BYTES",
        "PERPS_LOG_MAX_BYTES",
        "COINBASE_TRADER_LOG_BACKUP_COUNT",
        "PERPS_LOG_BACKUP_COUNT",
        "COINBASE_TRADER_DEBUG",
        "PERPS_DEBUG",
    ):
        monkeypatch.delenv(name, raising=False)
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(logging_setup, "LOG_DIR", log_dir)
    return log_dir


class TestConfigureLoggingEnvOverrides:
    """Test configure_logging env var overrides."""

    @pytest.fixture(autouse=True)
    def reset_logging(self) -> None:
        """Reset logging configuration before each test."""
        _clear_handlers()

        yield

        _clear_handlers()

    def test_configure_logging_respects_env_log_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that PERPS_LOG_DIR environment variable is respected."""
        custom_log_dir = tmp_path / "custom_logs"
        custom_log_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PERPS_LOG_DIR", str(custom_log_dir))

        configure_logging()

        # Check that file handlers were created in the custom directory
        rotating_handlers = _rotating_handlers(logging.getLogger())
        # At least one handler should use the custom log directory
        handler_paths = [Path(h.baseFilename).parent for h in rotating_handlers]
        assert custom_log_dir in handler_paths

    def test_configure_logging_respects_max_bytes_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that log size environment variables are respected."""
        custom_max_bytes = 1024 * 1024  # 1 MB
        monkeypatch.setenv("PERPS_LOG_MAX_BYTES", str(custom_max_bytes))

        configure_logging()

        rotating_handlers = _rotating_handlers(logging.getLogger()) + _rotating_handlers(
            logging.getLogger("gpt_trader.json")
        )

        # At least one handler should have the custom max bytes
        assert any(h.maxBytes == custom_max_bytes for h in rotating_handlers)

    def test_configure_logging_respects_backup_count_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that backup count environment variables are respected."""
        custom_backup_count = 20
        monkeypatch.setenv("PERPS_LOG_BACKUP_COUNT", str(custom_backup_count))

        configure_logging()

        rotating_handlers = _rotating_handlers(logging.getLogger()) + _rotating_handlers(
            logging.getLogger("gpt_trader.json")
        )

        # At least one handler should have the custom backup count
        assert any(h.backupCount == custom_backup_count for h in rotating_handlers)

    def test_configure_logging_enables_debug_with_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that PERPS_DEBUG environment variable enables debug logging."""
        monkeypatch.setenv("PERPS_DEBUG", "1")

        coinbase_logger = logging.getLogger("gpt_trader.features.brokerages.coinbase")
        live_trade_logger = logging.getLogger("gpt_trader.features.live_trade")
        original_levels = (coinbase_logger.level, live_trade_logger.level)

        try:
            configure_logging()

            # Check that specific loggers are set to DEBUG
            assert coinbase_logger.level == logging.DEBUG
            assert live_trade_logger.level == logging.DEBUG
        finally:
            coinbase_logger.setLevel(original_levels[0])
            live_trade_logger.setLevel(original_levels[1])

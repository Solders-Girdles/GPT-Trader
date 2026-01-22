"""Tests for configure_logging env overrides and mode behavior."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import pytest

import gpt_trader.logging.setup as logging_setup
from gpt_trader.logging.setup import configure_logging

_EXCLUDED_CONSOLE_HANDLER_TYPES = {"LogCaptureHandler", "LogCaptureFixture"}


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


def _console_handlers() -> list[logging.Handler]:
    root_logger = logging.getLogger()
    return [
        handler
        for handler in root_logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.handlers.RotatingFileHandler)
        and type(handler).__name__ not in _EXCLUDED_CONSOLE_HANDLER_TYPES
    ]


def _file_targets(logger: logging.Logger) -> list[str]:
    return [
        handler.baseFilename
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
        "COINBASE_TRADER_CRIT_LOG_MAX_BYTES",
        "PERPS_CRIT_LOG_MAX_BYTES",
        "COINBASE_TRADER_CRIT_LOG_BACKUP_COUNT",
        "PERPS_CRIT_LOG_BACKUP_COUNT",
        "COINBASE_TRADER_DEBUG",
        "PERPS_DEBUG",
    ):
        monkeypatch.delenv(name, raising=False)
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(logging_setup, "LOG_DIR", log_dir)
    return log_dir


@pytest.fixture(autouse=True)
def reset_logging() -> None:
    _clear_handlers()
    yield
    _clear_handlers()


class TestConfigureLoggingEnvOverrides:
    @pytest.mark.usefixtures("log_dir")
    def test_configure_logging_respects_env_log_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        custom_log_dir = tmp_path / "custom_logs"
        custom_log_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PERPS_LOG_DIR", str(custom_log_dir))

        configure_logging()

        rotating_handlers = _rotating_handlers(logging.getLogger())
        handler_paths = [Path(h.baseFilename).parent for h in rotating_handlers]
        assert custom_log_dir in handler_paths

    def test_configure_logging_respects_max_bytes_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        custom_max_bytes = 1024 * 1024
        monkeypatch.setenv("PERPS_LOG_MAX_BYTES", str(custom_max_bytes))

        configure_logging()

        rotating_handlers = _rotating_handlers(logging.getLogger()) + _rotating_handlers(
            logging.getLogger("gpt_trader.json")
        )

        assert any(h.maxBytes == custom_max_bytes for h in rotating_handlers)

    def test_configure_logging_respects_backup_count_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        custom_backup_count = 20
        monkeypatch.setenv("PERPS_LOG_BACKUP_COUNT", str(custom_backup_count))

        configure_logging()

        rotating_handlers = _rotating_handlers(logging.getLogger()) + _rotating_handlers(
            logging.getLogger("gpt_trader.json")
        )

        assert any(h.backupCount == custom_backup_count for h in rotating_handlers)

    def test_configure_logging_enables_debug_with_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("PERPS_DEBUG", "1")

        coinbase_logger = logging.getLogger("gpt_trader.features.brokerages.coinbase")
        live_trade_logger = logging.getLogger("gpt_trader.features.live_trade")
        original_levels = (coinbase_logger.level, live_trade_logger.level)

        try:
            configure_logging()

            assert coinbase_logger.level == logging.DEBUG
            assert live_trade_logger.level == logging.DEBUG
        finally:
            coinbase_logger.setLevel(original_levels[0])
            live_trade_logger.setLevel(original_levels[1])


class TestConfigureLoggingModesAndIdempotency:
    def test_configure_logging_idempotent(self, log_dir: Path) -> None:
        configure_logging()
        configure_logging()

        root_targets = _file_targets(logging.getLogger())
        json_targets = _file_targets(logging.getLogger("gpt_trader.json"))

        assert len(root_targets) == len(set(root_targets))
        assert len(json_targets) == len(set(json_targets))

    def test_configure_logging_tui_mode_suppresses_stream_handler(self, log_dir: Path) -> None:
        configure_logging(tui_mode=True)

        console_handlers = _console_handlers()

        assert len(console_handlers) == 0

    def test_configure_logging_cli_mode_creates_stream_handler(self, log_dir: Path) -> None:
        configure_logging(tui_mode=False)

        console_handlers = _console_handlers()

        assert len(console_handlers) >= 1

    def test_configure_logging_file_handlers_present_in_both_modes(self, log_dir: Path) -> None:
        expected_root = {"coinbase_trader.log", "critical_events.log"}
        expected_json = {"coinbase_trader.jsonl", "critical_events.jsonl"}

        configure_logging(tui_mode=True)

        root_targets = {Path(target).name for target in _file_targets(logging.getLogger())}
        json_targets = {
            Path(target).name for target in _file_targets(logging.getLogger("gpt_trader.json"))
        }
        assert expected_root.issubset(root_targets)
        assert expected_json.issubset(json_targets)

        _clear_handlers()

        configure_logging(tui_mode=False)

        cli_root_targets = {Path(target).name for target in _file_targets(logging.getLogger())}
        cli_json_targets = {
            Path(target).name for target in _file_targets(logging.getLogger("gpt_trader.json"))
        }
        assert expected_root.issubset(cli_root_targets)
        assert expected_json.issubset(cli_json_targets)
        assert root_targets == cli_root_targets
        assert json_targets == cli_json_targets

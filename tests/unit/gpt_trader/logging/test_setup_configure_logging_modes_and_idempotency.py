"""Tests for configure_logging behavior in TUI/CLI modes and idempotency."""

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


@pytest.fixture
def log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("COINBASE_TRADER_LOG_DIR", "PERPS_LOG_DIR"):
        monkeypatch.delenv(name, raising=False)
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(logging_setup, "LOG_DIR", log_dir)
    return log_dir


class TestConfigureLoggingModesAndIdempotency:
    """Test configure_logging mode behavior and idempotency."""

    @pytest.fixture(autouse=True)
    def reset_logging(self) -> None:
        """Reset logging configuration before each test."""
        _clear_handlers()

        yield

        _clear_handlers()

    def test_configure_logging_idempotent(self, log_dir: Path) -> None:
        """Test that calling configure_logging multiple times doesn't duplicate handlers."""
        configure_logging()
        configure_logging()

        root_targets = _file_targets(logging.getLogger())
        json_targets = _file_targets(logging.getLogger("gpt_trader.json"))

        assert len(root_targets) == len(set(root_targets))
        assert len(json_targets) == len(set(json_targets))

    def test_configure_logging_tui_mode_suppresses_stream_handler(self, log_dir: Path) -> None:
        """Test that TUI mode suppresses StreamHandler to prevent display corruption."""
        configure_logging(tui_mode=True)

        console_handlers = _console_handlers()

        # No console StreamHandler should be present in TUI mode
        assert len(console_handlers) == 0, "TUI mode should not have console StreamHandler"

    def test_configure_logging_cli_mode_creates_stream_handler(self, log_dir: Path) -> None:
        """Test that CLI mode creates StreamHandler for console output."""
        configure_logging(tui_mode=False)

        console_handlers = _console_handlers()

        # At least one console StreamHandler should be present in CLI mode
        assert len(console_handlers) >= 1, "CLI mode should have console StreamHandler"

    def test_configure_logging_file_handlers_present_in_both_modes(self, log_dir: Path) -> None:
        """Test that file handlers are present in both TUI and CLI modes."""
        expected_root = {"coinbase_trader.log", "critical_events.log"}
        expected_json = {"coinbase_trader.jsonl", "critical_events.jsonl"}

        # Test TUI mode
        configure_logging(tui_mode=True)

        root_targets = {Path(target).name for target in _file_targets(logging.getLogger())}
        json_targets = {
            Path(target).name for target in _file_targets(logging.getLogger("gpt_trader.json"))
        }
        assert expected_root.issubset(root_targets)
        assert expected_json.issubset(json_targets)

        # Reset handlers
        _clear_handlers()

        # Test CLI mode
        configure_logging(tui_mode=False)

        cli_root_targets = {Path(target).name for target in _file_targets(logging.getLogger())}
        cli_json_targets = {
            Path(target).name for target in _file_targets(logging.getLogger("gpt_trader.json"))
        }
        assert expected_root.issubset(cli_root_targets)
        assert expected_json.issubset(cli_json_targets)
        assert root_targets == cli_root_targets
        assert json_targets == cli_json_targets

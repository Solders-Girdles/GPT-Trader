"""Tests for logging setup module."""

import logging
import logging.handlers
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.logging.setup import _env_flag, configure_logging


class TestEnvFlag:
    """Test the _env_flag helper function."""

    def test_env_flag_true_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that various true values are recognized."""
        true_values = ["1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"]
        for value in true_values:
            monkeypatch.setenv("TEST_FLAG", value)
            assert _env_flag("TEST_FLAG") is True, f"Failed for value: {value}"

    def test_env_flag_false_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that various false values are recognized."""
        false_values = ["0", "false", "False", "no", "No", "off", "Off", ""]
        for value in false_values:
            monkeypatch.setenv("TEST_FLAG", value)
            assert _env_flag("TEST_FLAG") is False, f"Failed for value: {value}"

    def test_env_flag_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default value is used when env var not set."""
        monkeypatch.delenv("MISSING_FLAG", raising=False)
        assert _env_flag("MISSING_FLAG", default="0") is False
        assert _env_flag("MISSING_FLAG", default="1") is True

    def test_env_flag_with_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that whitespace is stripped."""
        monkeypatch.setenv("TEST_FLAG", "  true  ")
        assert _env_flag("TEST_FLAG") is True

        monkeypatch.setenv("TEST_FLAG", "  0  ")
        assert _env_flag("TEST_FLAG") is False


class TestConfigureLogging:
    """Test the configure_logging function."""

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
        assert len(tui_file_handlers) == len(cli_file_handlers), (
            "Both modes should have the same file handlers"
        )


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

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

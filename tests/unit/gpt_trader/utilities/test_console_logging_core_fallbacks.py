"""
Tests for console logging utilities - fallback behavior and global helpers.
"""

import logging
from io import StringIO
from unittest.mock import Mock

import pytest

import gpt_trader.utilities.console_logging as console_logging_module
from gpt_trader.utilities.console_logging import ConsoleLogger, get_console_logger


class TestConsoleLoggerFallbacks:
    @pytest.fixture(autouse=True)
    def reset_console_logger_singleton(self):
        console_logging_module._console_logger = None
        yield
        console_logging_module._console_logger = None

    @pytest.fixture
    def mock_output_stream(self):
        return StringIO()

    @pytest.fixture
    def console_logger(self, mock_output_stream):
        return ConsoleLogger(enable_console=True, output_stream=mock_output_stream)

    def test_structured_logging_integration(self, console_logger, caplog):
        caplog.set_level(logging.INFO, logger="gpt_trader.utilities.console_logging")

        console_logger.success("Test message", test_param="value")

        assert any(
            record.name == "gpt_trader.utilities.console_logging"
            and "Test message" in record.getMessage()
            for record in caplog.records
        )

    def test_output_stream_error_handling(self, mock_output_stream):
        mock_output_stream.write = Mock(side_effect=Exception("Stream error"))

        logger = ConsoleLogger(enable_console=True, output_stream=mock_output_stream)

        logger.success("Test message")

        mock_output_stream.write.assert_called()

    def test_print_section_fallback_on_failure(self, monkeypatch):
        recorded: list[str] = []
        failing_stream = object()

        def fake_print(*args, **kwargs):
            if kwargs.get("file") is failing_stream:
                raise Exception("stream failure")
            recorded.append(args[0])

        monkeypatch.setattr("builtins.print", fake_print)
        logger = ConsoleLogger(enable_console=True, output_stream=failing_stream)

        logger.print_section("Fallback", "#", 10)

        assert recorded
        assert "Fallback" in recorded[0]

    def test_print_table_fallback_on_failure(self, monkeypatch):
        recorded: list[str] = []
        failing_stream = object()

        def fake_print(*args, **kwargs):
            if kwargs.get("file") is failing_stream:
                raise Exception("stream failure")
            recorded.append(args[0])

        monkeypatch.setattr("builtins.print", fake_print)
        logger = ConsoleLogger(enable_console=True, output_stream=failing_stream)

        headers = ["Name"]
        rows = [["Alice"], ["Bob"]]
        logger.print_table(headers, rows)

        assert recorded[0].strip() == "Name"
        assert recorded[1].strip().startswith("-")
        assert recorded[2].strip().startswith("Alice")
        assert recorded[3].strip().startswith("Bob")

    def test_print_key_value_fallback_on_failure(self, monkeypatch):
        recorded: list[str] = []
        failing_stream = object()

        def fake_print(*args, **kwargs):
            if kwargs.get("file") is failing_stream:
                raise Exception("stream failure")
            recorded.append(args[0])

        monkeypatch.setattr("builtins.print", fake_print)
        logger = ConsoleLogger(enable_console=True, output_stream=failing_stream)

        logger.printKeyValue("Key", "Value", indent=1)

        assert recorded == ["   Key: Value"]


class TestGlobalConsoleLogger:
    @pytest.fixture(autouse=True)
    def reset_console_logger_singleton(self):
        console_logging_module._console_logger = None
        yield
        console_logging_module._console_logger = None

    def test_get_console_logger_singleton(self):
        logger1 = get_console_logger()
        logger2 = get_console_logger()

        assert logger1 is logger2

    def test_get_console_logger_custom_settings(self):
        logger = get_console_logger(enable_console=False)

        assert logger.enable_console is False

    def test_get_console_logger_preserves_existing_instance(self):
        first = get_console_logger(enable_console=True)
        first.enable_console = True

        second = get_console_logger(enable_console=False)
        assert second is first
        assert second.enable_console is True

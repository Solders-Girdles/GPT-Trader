"""
Tests for console logging utilities - Core functionality.

This module tests the core ConsoleLogger class functionality.
"""

import pytest
from unittest.mock import Mock, patch
from io import StringIO
import sys

from bot_v2.utilities.console_logging import (
    ConsoleLogger,
    get_console_logger,
)


class TestConsoleLogger:
    """Test cases for ConsoleLogger class."""

    @pytest.fixture
    def mock_output_stream(self):
        """Create a mock output stream."""
        return StringIO()

    @pytest.fixture
    def console_logger(self, mock_output_stream):
        """Create ConsoleLogger instance with mock output."""
        return ConsoleLogger(enable_console=True, output_stream=mock_output_stream)

    def test_init_default(self):
        """Test ConsoleLogger initialization with defaults."""
        logger = ConsoleLogger()

        assert logger.enable_console is True
        assert logger.output_stream == sys.stdout

    def test_init_custom(self, mock_output_stream):
        """Test ConsoleLogger initialization with custom settings."""
        logger = ConsoleLogger(enable_console=False, output_stream=mock_output_stream)

        assert logger.enable_console is False
        assert logger.output_stream == mock_output_stream

    def test_success_logging(self, console_logger, mock_output_stream):
        """Test success message logging."""
        console_logger.success("Test success", test_id="123")

        output = mock_output_stream.getvalue()
        assert "✅ Test success" in output

    def test_error_logging(self, console_logger, mock_output_stream):
        """Test error message logging."""
        console_logger.error("Test error", error_code="E001")

        output = mock_output_stream.getvalue()
        assert "❌ Test error" in output

    def test_warning_logging(self, console_logger, mock_output_stream):
        """Test warning message logging."""
        console_logger.warning("Test warning", warning_type="performance")

        output = mock_output_stream.getvalue()
        assert "⚠️ Test warning" in output

    def test_info_logging(self, console_logger, mock_output_stream):
        """Test info message logging."""
        console_logger.info("Test info", info_type="general")

        output = mock_output_stream.getvalue()
        assert "ℹ️ Test info" in output

    def test_context_specific_logging(self, console_logger, mock_output_stream):
        """Test context-specific logging methods."""
        # Test data logging
        console_logger.data("Data processed", records=100)
        output = mock_output_stream.getvalue()
        assert "📊 Data processed" in output

        # Clear output
        mock_output_stream.seek(0)
        mock_output_stream.truncate(0)

        # Test trading logging
        console_logger.trading("Trade executed", symbol="BTC-USD")
        output = mock_output_stream.getvalue()
        assert "💰 Trade executed" in output

        # Clear output
        mock_output_stream.seek(0)
        mock_output_stream.truncate(0)

        # Test order logging
        console_logger.order("Order placed", order_id="123")
        output = mock_output_stream.getvalue()
        assert "📝 Order placed" in output

        # Clear output
        mock_output_stream.seek(0)
        mock_output_stream.truncate(0)

        # Test position logging
        console_logger.position("Position updated", symbol="ETH-USD")
        output = mock_output_stream.getvalue()
        assert "📈 Position updated" in output

    def test_disabled_console_logging(self, mock_output_stream):
        """Test logging when console is disabled."""
        logger = ConsoleLogger(enable_console=False, output_stream=mock_output_stream)

        logger.success("This should not appear")

        output = mock_output_stream.getvalue()
        assert output == ""

    def test_all_methods_when_console_disabled(self, mock_output_stream):
        """Ensure all console methods skip printing when disabled."""
        logger = ConsoleLogger(enable_console=False, output_stream=mock_output_stream)
        for method_name in [
            "success",
            "error",
            "warning",
            "info",
            "data",
            "trading",
            "order",
            "position",
            "cache",
            "storage",
            "network",
            "analysis",
            "ml",
        ]:
            getattr(logger, method_name)("message")

        assert mock_output_stream.getvalue() == ""

    def test_print_section(self, console_logger, mock_output_stream):
        """Test section printing."""
        console_logger.print_section("Test Section", "=", 30)

        output = mock_output_stream.getvalue()
        assert "Test Section" in output
        assert "=" in output

    def test_print_section_no_title(self, console_logger, mock_output_stream):
        """Test section printing without title."""
        console_logger.print_section("", "-", 20)

        output = mock_output_stream.getvalue()
        assert "-" * 20 in output

    def test_print_section_disabled(self, mock_output_stream):
        """Section printing does nothing when console is disabled."""
        logger = ConsoleLogger(enable_console=False, output_stream=mock_output_stream)
        logger.print_section("Hidden", "*", 10)
        assert mock_output_stream.getvalue() == ""

    def test_print_table(self, console_logger, mock_output_stream):
        """Test table printing."""
        headers = ["Symbol", "Price", "Quantity"]
        rows = [
            ["BTC-USD", "50000", "1.0"],
            ["ETH-USD", "3000", "10.0"],
            # Extra column to exercise branch where i >= len(col_widths)
            ["SOL-USD", "150", "20.0", "ignored"],
        ]

        console_logger.print_table(headers, rows)

        output = mock_output_stream.getvalue()
        assert "Symbol" in output
        assert "Price" in output
        assert "Quantity" in output
        assert "BTC-USD" in output
        assert "ETH-USD" in output
        assert "-" in output  # Separator line

    def test_print_table_empty_rows(self, console_logger, mock_output_stream):
        """Test table printing with empty rows."""
        headers = ["Symbol", "Price"]
        rows = []

        console_logger.print_table(headers, rows)

        output = mock_output_stream.getvalue()
        assert output == ""

    def test_print_table_disabled(self, mock_output_stream):
        """Test table printing when console is disabled."""
        logger = ConsoleLogger(enable_console=False, output_stream=mock_output_stream)

        headers = ["Symbol", "Price"]
        rows = [["BTC-USD", "50000"]]

        logger.print_table(headers, rows)

        output = mock_output_stream.getvalue()
        assert output == ""

    def test_print_key_value(self, console_logger, mock_output_stream):
        """Test key-value pair printing."""
        console_logger.printKeyValue("Test Key", "Test Value")

        output = mock_output_stream.getvalue()
        assert "Test Key: Test Value" in output

    def test_print_key_value_with_indent(self, console_logger, mock_output_stream):
        """Test key-value pair printing with indentation."""
        console_logger.printKeyValue("Nested Key", "Nested Value", indent=2)

        output = mock_output_stream.getvalue()
        assert "      Nested Key: Nested Value" in output  # 6 spaces (2 * 3)

    def test_print_key_value_disabled(self, mock_output_stream):
        """Test key-value printing when console is disabled."""
        logger = ConsoleLogger(enable_console=False, output_stream=mock_output_stream)

        logger.printKeyValue("Test Key", "Test Value")

        output = mock_output_stream.getvalue()
        assert output == ""

    @patch("bot_v2.utilities.console_logging.logger")
    def test_structured_logging_integration(self, mock_structured_logger, console_logger):
        """Test integration with structured logging."""
        console_logger.success("Test message", test_param="value")

        # Verify structured logger was called
        mock_structured_logger.info.assert_called_once_with("Test message", test_param="value")

    def test_output_stream_error_handling(self, mock_output_stream):
        """Test handling of output stream errors."""
        # Make the output stream raise an exception
        mock_output_stream.write = Mock(side_effect=Exception("Stream error"))

        logger = ConsoleLogger(enable_console=True, output_stream=mock_output_stream)

        # Should not raise exception due to fallback
        logger.success("Test message")

        # Verify write was attempted
        mock_output_stream.write.assert_called()

    def test_print_section_fallback_on_failure(self, monkeypatch):
        """Fallback should print without stream when section print fails."""
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
        """Fallback branch should execute when printing rows fails."""
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
        """printKeyValue should fall back to default stdout when stream fails."""
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
    """Test cases for global console logger functions."""

    def test_get_console_logger_singleton(self):
        """Test that get_console_logger returns the same instance."""
        logger1 = get_console_logger()
        logger2 = get_console_logger()

        assert logger1 is logger2

    def test_get_console_logger_custom_settings(self):
        """Test get_console_logger with custom settings."""
        # Reset global instance
        import bot_v2.utilities.console_logging

        bot_v2.utilities.console_logging._console_logger = None

        logger = get_console_logger(enable_console=False)

        assert logger.enable_console is False

    def test_get_console_logger_preserves_existing_instance(self):
        """Calling get_console_logger after initialization should reuse singleton."""
        import bot_v2.utilities.console_logging

        bot_v2.utilities.console_logging._console_logger = None
        first = get_console_logger(enable_console=True)
        first.enable_console = True

        second = get_console_logger(enable_console=False)
        assert second is first
        # The original enable_console flag should remain unchanged
        assert second.enable_console is True

"""
Tests for console logging utilities - Core functionality.

This module covers the core ConsoleLogger output behavior and printing helpers.
"""

import sys
from io import StringIO

import pytest

from gpt_trader.utilities.console_logging import ConsoleLogger


class TestConsoleLoggerCore:
    @pytest.fixture
    def mock_output_stream(self):
        return StringIO()

    @pytest.fixture
    def console_logger(self, mock_output_stream):
        return ConsoleLogger(enable_console=True, output_stream=mock_output_stream)

    def test_init_default(self):
        logger = ConsoleLogger()

        assert logger.enable_console is True
        assert logger.output_stream == sys.stdout

    def test_init_custom(self, mock_output_stream):
        logger = ConsoleLogger(enable_console=False, output_stream=mock_output_stream)

        assert logger.enable_console is False
        assert logger.output_stream == mock_output_stream

    def test_success_logging(self, console_logger, mock_output_stream):
        console_logger.success("Test success", test_id="123")

        output = mock_output_stream.getvalue()
        assert "✅ Test success" in output

    def test_error_logging(self, console_logger, mock_output_stream):
        console_logger.error("Test error", error_code="E001")

        output = mock_output_stream.getvalue()
        assert "❌ Test error" in output

    def test_warning_logging(self, console_logger, mock_output_stream):
        console_logger.warning("Test warning", warning_type="performance")

        output = mock_output_stream.getvalue()
        assert "⚠️ Test warning" in output

    def test_info_logging(self, console_logger, mock_output_stream):
        console_logger.info("Test info", info_type="general")

        output = mock_output_stream.getvalue()
        assert "ℹ️ Test info" in output

    def test_context_specific_logging(self, console_logger, mock_output_stream):
        console_logger.data("Data processed", records=100)
        output = mock_output_stream.getvalue()
        assert "📊 Data processed" in output

        mock_output_stream.seek(0)
        mock_output_stream.truncate(0)

        console_logger.trading("Trade executed", symbol="BTC-USD")
        output = mock_output_stream.getvalue()
        assert "💰 Trade executed" in output

        mock_output_stream.seek(0)
        mock_output_stream.truncate(0)

        console_logger.order("Order placed", order_id="123")
        output = mock_output_stream.getvalue()
        assert "📝 Order placed" in output

        mock_output_stream.seek(0)
        mock_output_stream.truncate(0)

        console_logger.position("Position updated", symbol="ETH-USD")
        output = mock_output_stream.getvalue()
        assert "📈 Position updated" in output

    def test_disabled_console_logging(self, mock_output_stream):
        logger = ConsoleLogger(enable_console=False, output_stream=mock_output_stream)

        logger.success("This should not appear")

        output = mock_output_stream.getvalue()
        assert output == ""

    def test_all_methods_when_console_disabled(self, mock_output_stream):
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
        console_logger.print_section("Test Section", "=", 30)

        output = mock_output_stream.getvalue()
        assert "Test Section" in output
        assert "=" in output

    def test_print_section_no_title(self, console_logger, mock_output_stream):
        console_logger.print_section("", "-", 20)

        output = mock_output_stream.getvalue()
        assert "-" * 20 in output

    def test_print_section_disabled(self, mock_output_stream):
        logger = ConsoleLogger(enable_console=False, output_stream=mock_output_stream)
        logger.print_section("Hidden", "*", 10)
        assert mock_output_stream.getvalue() == ""

    def test_print_table(self, console_logger, mock_output_stream):
        headers = ["Symbol", "Price", "Quantity"]
        rows = [
            ["BTC-USD", "50000", "1.0"],
            ["ETH-USD", "3000", "10.0"],
            ["SOL-USD", "150", "20.0", "ignored"],
        ]

        console_logger.print_table(headers, rows)

        output = mock_output_stream.getvalue()
        assert "Symbol" in output
        assert "Price" in output
        assert "Quantity" in output
        assert "BTC-USD" in output
        assert "ETH-USD" in output
        assert "-" in output

    def test_print_table_empty_rows(self, console_logger, mock_output_stream):
        headers = ["Symbol", "Price"]
        rows = []

        console_logger.print_table(headers, rows)

        output = mock_output_stream.getvalue()
        assert output == ""

    def test_print_table_disabled(self, mock_output_stream):
        logger = ConsoleLogger(enable_console=False, output_stream=mock_output_stream)

        headers = ["Symbol", "Price"]
        rows = [["BTC-USD", "50000"]]

        logger.print_table(headers, rows)

        output = mock_output_stream.getvalue()
        assert output == ""

    def test_print_key_value(self, console_logger, mock_output_stream):
        console_logger.printKeyValue("Test Key", "Test Value")

        output = mock_output_stream.getvalue()
        assert "Test Key: Test Value" in output

    def test_print_key_value_with_indent(self, console_logger, mock_output_stream):
        console_logger.printKeyValue("Nested Key", "Nested Value", indent=2)

        output = mock_output_stream.getvalue()
        assert "      Nested Key: Nested Value" in output

    def test_print_key_value_disabled(self, mock_output_stream):
        logger = ConsoleLogger(enable_console=False, output_stream=mock_output_stream)

        logger.printKeyValue("Test Key", "Test Value")

        output = mock_output_stream.getvalue()
        assert output == ""

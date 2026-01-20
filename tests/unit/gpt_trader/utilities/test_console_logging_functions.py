"""Tests for console logging global helpers."""

from unittest.mock import Mock

import pytest

import gpt_trader.utilities.console_logging as console_logging_module
from gpt_trader.utilities.console_logging import (
    console_cache,
    console_data,
    console_error,
    console_info,
    console_order,
    console_position,
    console_section,
    console_success,
    console_table,
    console_trading,
    console_warning,
)


@pytest.fixture
def console_logger_mock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock_logger = Mock()
    monkeypatch.setattr(console_logging_module, "ConsoleLogger", Mock(return_value=mock_logger))
    return mock_logger


@pytest.fixture
def get_console_logger_mock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock_logger = Mock()
    monkeypatch.setattr(
        console_logging_module, "get_console_logger", Mock(return_value=mock_logger)
    )
    return mock_logger


class TestConsoleLoggingFunctions:
    """Tests for console logging utilities."""

    def test_end_to_end_logging_workflow(self, console_logger_mock: Mock):
        """Test complete logging workflow."""
        console_logging_module._console_logger = None

        console_section("Trading Session")
        console_success("Connected to broker")
        console_order("Order placed", symbol="BTC-USD", quantity=1.0)
        console_position("Position updated", symbol="BTC-USD", quantity=1.0)
        console_data("Market data received", symbols=["BTC-USD", "ETH-USD"])

        console_logging_module.ConsoleLogger.assert_called_once_with(enable_console=True)
        assert console_logging_module._console_logger is console_logger_mock

        console_logger_mock.print_section.assert_called_once_with("Trading Session", "=", 50)
        console_logger_mock.success.assert_called_once_with("Connected to broker")
        console_logger_mock.order.assert_called_once_with(
            "Order placed", symbol="BTC-USD", quantity=1.0
        )
        console_logger_mock.position.assert_called_once_with(
            "Position updated", symbol="BTC-USD", quantity=1.0
        )
        console_logger_mock.data.assert_called_once_with(
            "Market data received", symbols=["BTC-USD", "ETH-USD"]
        )

    def test_all_context_methods(self):
        """Test all context-specific methods work without errors."""
        console_logging_module._console_logger = None

        methods = [
            console_data,
            console_trading,
            console_order,
            console_position,
            console_cache,
        ]

        for method in methods:
            try:
                method("Test message")
            except Exception as e:
                pytest.fail(f"Method {method.__name__} raised exception: {e}")

    def test_table_formatting_complex_data(self, console_logger_mock: Mock):
        """Test table formatting with complex data."""
        console_logging_module._console_logger = None

        headers = ["Symbol", "Price", "Quantity", "Value", "P&L", "P&L %"]
        rows = [
            ["BTC-USD", "50,123.45", "1.5", "75,185.18", "+1,234.56", "+1.67%"],
            ["ETH-USD", "3,456.78", "10.2", "35,259.16", "-123.45", "-0.35%"],
            ["SOL-USD", "123.45", "100.0", "12,345.00", "+567.89", "+4.81%"],
        ]

        console_table(headers, rows)
        console_table(headers, [])
        single_row = [rows[0]]
        console_table(headers, single_row)

        assert console_logger_mock.print_table.call_count == 3
        assert console_logger_mock.print_table.call_args_list[0].args == (headers, rows)
        assert console_logger_mock.print_table.call_args_list[1].args == (headers, [])
        assert console_logger_mock.print_table.call_args_list[2].args == (headers, single_row)

    def test_error_handling_in_global_functions(self):
        """Test error handling in global console functions."""
        console_logging_module._console_logger = None

        try:
            console_success("Test message")
            console_error("Test error")
            console_warning("Test warning")
            console_info("Test info")
        except Exception as e:
            pytest.fail(f"Global console functions raised exception: {e}")

    def test_function_parameter_passing(self, get_console_logger_mock: Mock):
        """Test that parameters are correctly passed to underlying logger."""
        console_success("Test", string_param="value", int_param=42, float_param=3.14)
        get_console_logger_mock.success.assert_called_once_with(
            "Test", string_param="value", int_param=42, float_param=3.14
        )

        get_console_logger_mock.reset_mock()
        console_error("Error", exception=ValueError("test"), none_param=None)
        get_console_logger_mock.error.assert_called_once()
        call_args = get_console_logger_mock.error.call_args
        assert call_args[0][0] == "Error"
        assert "exception" in call_args[1]
        assert "none_param" in call_args[1]
        assert call_args[1]["none_param"] is None

    def test_multiple_function_calls(self, get_console_logger_mock: Mock):
        """Test multiple calls to console functions."""
        console_success("Success 1")
        console_error("Error 1")
        console_warning("Warning 1")
        console_info("Info 1")

        assert get_console_logger_mock.success.call_count == 1
        assert get_console_logger_mock.error.call_count == 1
        assert get_console_logger_mock.warning.call_count == 1
        assert get_console_logger_mock.info.call_count == 1

        expected_calls = [
            (("Success 1",), {}),
            (("Error 1",), {}),
            (("Warning 1",), {}),
            (("Info 1",), {}),
        ]

        actual_calls = []
        for method_name in ["success", "error", "warning", "info"]:
            method = getattr(get_console_logger_mock, method_name)
            if method.called:
                actual_calls.append((method.call_args[0], method.call_args[1]))

        assert actual_calls == expected_calls

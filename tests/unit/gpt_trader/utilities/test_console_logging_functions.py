"""
Tests for console logging utilities - Global functions and integration.

This module tests the global console logging functions and integration scenarios.
"""

from unittest.mock import Mock, patch

import pytest

from gpt_trader.utilities.console_logging import (
    console_analysis,
    console_cache,
    console_data,
    console_error,
    console_info,
    console_key_value,
    console_ml,
    console_network,
    console_order,
    console_position,
    console_section,
    console_storage,
    console_success,
    console_table,
    console_trading,
    console_warning,
)


class TestGlobalConsoleFunctions:
    """Test cases for global console logger functions."""

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_console_success_function(self, mock_get_logger):
        """Test console_success function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        console_success("Test success", test_id="123")

        mock_logger.success.assert_called_once_with("Test success", test_id="123")

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_console_error_function(self, mock_get_logger):
        """Test console_error function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        console_error("Test error", error_code="E001")

        mock_logger.error.assert_called_once_with("Test error", error_code="E001")

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_console_warning_function(self, mock_get_logger):
        """Test console_warning function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        console_warning("Test warning", warning_type="performance")

        mock_logger.warning.assert_called_once_with("Test warning", warning_type="performance")

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_console_info_function(self, mock_get_logger):
        """Test console_info function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        console_info("Test info", info_type="general")

        mock_logger.info.assert_called_once_with("Test info", info_type="general")

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_context_specific_functions(self, mock_get_logger):
        """Test context-specific console functions."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Test data function
        console_data("Test data", records=100)
        mock_logger.data.assert_called_once_with("Test data", records=100)

        # Test trading function
        mock_logger.reset_mock()
        console_trading("Test trading", symbol="BTC-USD")
        mock_logger.trading.assert_called_once_with("Test trading", symbol="BTC-USD")

        # Test order function
        mock_logger.reset_mock()
        console_order("Test order", order_id="123")
        mock_logger.order.assert_called_once_with("Test order", order_id="123")

        # Test position function
        mock_logger.reset_mock()
        console_position("Test position", symbol="ETH-USD")
        mock_logger.position.assert_called_once_with("Test position", symbol="ETH-USD")

        # Test cache function
        mock_logger.reset_mock()
        console_cache("Test cache", cache_key="test_key")
        mock_logger.cache.assert_called_once_with("Test cache", cache_key="test_key")

        # Test storage function
        mock_logger.reset_mock()
        console_storage("Test storage", file="test.json")
        mock_logger.storage.assert_called_once_with("Test storage", file="test.json")

        # Test network function
        mock_logger.reset_mock()
        console_network("Test network", endpoint="api.test.com")
        mock_logger.network.assert_called_once_with("Test network", endpoint="api.test.com")

        # Test analysis function
        mock_logger.reset_mock()
        console_analysis("Test analysis", metric="sharpe")
        mock_logger.analysis.assert_called_once_with("Test analysis", metric="sharpe")

        # Test ML function
        mock_logger.reset_mock()
        console_ml("Test ML", model="test_model")
        mock_logger.ml.assert_called_once_with("Test ML", model="test_model")

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_console_section_function(self, mock_get_logger):
        """Test console_section function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        console_section("Test Section", "=", 40)

        mock_logger.print_section.assert_called_once_with("Test Section", "=", 40)

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_console_table_function(self, mock_get_logger):
        """Test console_table function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        headers = ["Symbol", "Price"]
        rows = [["BTC-USD", "50000"]]

        console_table(headers, rows)

        mock_logger.print_table.assert_called_once_with(headers, rows)

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_console_key_value_function(self, mock_get_logger):
        """Test console_key_value function."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        console_key_value("Test Key", "Test Value", 2)

        mock_logger.printKeyValue.assert_called_once_with("Test Key", "Test Value", 2)


class TestIntegration:
    """Integration tests for console logging utilities."""

    def test_end_to_end_logging_workflow(self):
        """Test complete logging workflow."""
        # Reset global instance
        import gpt_trader.utilities.console_logging

        gpt_trader.utilities.console_logging._console_logger = None

        mock_logger = Mock()
        with patch(
            "gpt_trader.utilities.console_logging.ConsoleLogger", return_value=mock_logger
        ) as mock_console:
            # Use global functions
            console_section("Trading Session")
            console_success("Connected to broker")
            console_order("Order placed", symbol="BTC-USD", quantity=1.0)
            console_position("Position updated", symbol="BTC-USD", quantity=1.0)
            console_data("Market data received", symbols=["BTC-USD", "ETH-USD"])

            mock_console.assert_called_once_with(enable_console=True)
            assert gpt_trader.utilities.console_logging._console_logger is mock_logger

            mock_logger.print_section.assert_called_once_with("Trading Session", "=", 50)
            mock_logger.success.assert_called_once_with("Connected to broker")
            mock_logger.order.assert_called_once_with(
                "Order placed", symbol="BTC-USD", quantity=1.0
            )
            mock_logger.position.assert_called_once_with(
                "Position updated", symbol="BTC-USD", quantity=1.0
            )
            mock_logger.data.assert_called_once_with(
                "Market data received", symbols=["BTC-USD", "ETH-USD"]
            )

    def test_all_context_methods(self):
        """Test all context-specific methods work without errors."""
        # Reset global instance
        import gpt_trader.utilities.console_logging

        gpt_trader.utilities.console_logging._console_logger = None

        # Test all methods
        methods = [
            console_data,
            console_trading,
            console_order,
            console_position,
            console_cache,
            console_storage,
            console_network,
            console_analysis,
            console_ml,
        ]

        for method in methods:
            try:
                method("Test message")
            except Exception as e:
                pytest.fail(f"Method {method.__name__} raised exception: {e}")

    def test_table_formatting_complex_data(self):
        """Test table formatting with complex data."""
        # Reset global instance
        import gpt_trader.utilities.console_logging

        gpt_trader.utilities.console_logging._console_logger = None

        headers = ["Symbol", "Price", "Quantity", "Value", "P&L", "P&L %"]
        rows = [
            ["BTC-USD", "50,123.45", "1.5", "75,185.18", "+1,234.56", "+1.67%"],
            ["ETH-USD", "3,456.78", "10.2", "35,259.16", "-123.45", "-0.35%"],
            ["SOL-USD", "123.45", "100.0", "12,345.00", "+567.89", "+4.81%"],
        ]

        # Should not raise any exceptions
        console_table(headers, rows)

        # Test with empty rows
        console_table(headers, [])

        # Test with single row
        single_row = [rows[0]]
        console_table(headers, single_row)

    def test_error_handling_in_global_functions(self):
        """Test error handling in global console functions."""
        # Reset global instance
        import gpt_trader.utilities.console_logging

        gpt_trader.utilities.console_logging._console_logger = None

        # Test that functions handle errors gracefully
        # This is mainly to ensure no uncaught exceptions
        try:
            console_success("Test message")
            console_error("Test error")
            console_warning("Test warning")
            console_info("Test info")
        except Exception as e:
            pytest.fail(f"Global console functions raised exception: {e}")

    def test_function_parameter_passing(self):
        """Test that parameters are correctly passed to underlying logger."""
        with patch("gpt_trader.utilities.console_logging.get_console_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # Test with various parameter types
            console_success("Test", string_param="value", int_param=42, float_param=3.14)
            mock_logger.success.assert_called_once_with(
                "Test", string_param="value", int_param=42, float_param=3.14
            )

            mock_logger.reset_mock()
            console_error("Error", exception=ValueError("test"), none_param=None)
            # Verify the call was made with correct parameters (ValueError instances can't be compared directly)
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert call_args[0][0] == "Error"
            assert "exception" in call_args[1]
            assert "none_param" in call_args[1]
            assert call_args[1]["none_param"] is None

    def test_multiple_function_calls(self):
        """Test multiple calls to console functions."""
        with patch("gpt_trader.utilities.console_logging.get_console_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # Make multiple calls
            console_success("Success 1")
            console_error("Error 1")
            console_warning("Warning 1")
            console_info("Info 1")

            # Verify all calls were made
            assert mock_logger.success.call_count == 1
            assert mock_logger.error.call_count == 1
            assert mock_logger.warning.call_count == 1
            assert mock_logger.info.call_count == 1

            # Verify call order
            expected_calls = [
                (("Success 1",), {}),
                (("Error 1",), {}),
                (("Warning 1",), {}),
                (("Info 1",), {}),
            ]

            actual_calls = []
            for method_name in ["success", "error", "warning", "info"]:
                method = getattr(mock_logger, method_name)
                if method.called:
                    actual_calls.append((method.call_args[0], method.call_args[1]))

            assert actual_calls == expected_calls

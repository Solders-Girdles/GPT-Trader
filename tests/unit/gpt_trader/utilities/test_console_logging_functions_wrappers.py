"""Tests for console logging utilities - Global wrapper functions."""

from unittest.mock import Mock, patch

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
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        console_success("Test success", test_id="123")

        mock_logger.success.assert_called_once_with("Test success", test_id="123")

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_console_error_function(self, mock_get_logger):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        console_error("Test error", error_code="E001")

        mock_logger.error.assert_called_once_with("Test error", error_code="E001")

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_console_warning_function(self, mock_get_logger):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        console_warning("Test warning", warning_type="performance")

        mock_logger.warning.assert_called_once_with("Test warning", warning_type="performance")

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_console_info_function(self, mock_get_logger):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        console_info("Test info", info_type="general")

        mock_logger.info.assert_called_once_with("Test info", info_type="general")

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_context_specific_functions(self, mock_get_logger):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        console_data("Test data", records=100)
        mock_logger.data.assert_called_once_with("Test data", records=100)

        mock_logger.reset_mock()
        console_trading("Test trading", symbol="BTC-USD")
        mock_logger.trading.assert_called_once_with("Test trading", symbol="BTC-USD")

        mock_logger.reset_mock()
        console_order("Test order", order_id="123")
        mock_logger.order.assert_called_once_with("Test order", order_id="123")

        mock_logger.reset_mock()
        console_position("Test position", symbol="ETH-USD")
        mock_logger.position.assert_called_once_with("Test position", symbol="ETH-USD")

        mock_logger.reset_mock()
        console_cache("Test cache", cache_key="test_key")
        mock_logger.cache.assert_called_once_with("Test cache", cache_key="test_key")

        mock_logger.reset_mock()
        console_storage("Test storage", file="test.json")
        mock_logger.storage.assert_called_once_with("Test storage", file="test.json")

        mock_logger.reset_mock()
        console_network("Test network", endpoint="api.test.com")
        mock_logger.network.assert_called_once_with("Test network", endpoint="api.test.com")

        mock_logger.reset_mock()
        console_analysis("Test analysis", metric="sharpe")
        mock_logger.analysis.assert_called_once_with("Test analysis", metric="sharpe")

        mock_logger.reset_mock()
        console_ml("Test ML", model="test_model")
        mock_logger.ml.assert_called_once_with("Test ML", model="test_model")

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_console_section_function(self, mock_get_logger):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        console_section("Test Section", "=", 40)

        mock_logger.print_section.assert_called_once_with("Test Section", "=", 40)

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_console_table_function(self, mock_get_logger):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        headers = ["Symbol", "Price"]
        rows = [["BTC-USD", "50000"]]

        console_table(headers, rows)

        mock_logger.print_table.assert_called_once_with(headers, rows)

    @patch("gpt_trader.utilities.console_logging.get_console_logger")
    def test_console_key_value_function(self, mock_get_logger):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        console_key_value("Test Key", "Test Value", 2)

        mock_logger.printKeyValue.assert_called_once_with("Test Key", "Test Value", 2)

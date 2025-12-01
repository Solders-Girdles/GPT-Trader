import logging
from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.widgets.logs import LogWidget


@pytest.mark.asyncio
async def test_data_flow_from_reporter_to_state(mock_app):
    """Verify that status reporter updates are correctly applied to TuiState."""
    # Simulate status update
    status_update = {
        "system": {
            "api_latency": 123.45,
            "connection_status": "CONNECTED",
            "memory_usage": "512MB",
            "cpu_usage": "15%",
        },
        "market": {"last_prices": {"BTC-USD": "50000.00"}, "last_price_update": 1000000.0},
        "account": {"balances": [{"asset": "USD", "total": "1000.00", "available": "500.00"}]},
    }

    # Apply update directly (bypassing thread scheduling for test simplicity)
    mock_app._apply_status_update(status_update)

    # Verify State
    assert mock_app.tui_state.system_data.api_latency == 123.45
    assert mock_app.tui_state.system_data.connection_status == "CONNECTED"
    assert mock_app.tui_state.market_data.prices["BTC-USD"] == "50000.00"
    # AccountBalance is a dataclass, access via attributes
    assert mock_app.tui_state.account_data.balances[0].asset == "USD"


@pytest.mark.asyncio
async def test_log_filtering(mock_app):
    """Verify LogWidget filtering logic."""
    widget = LogWidget()
    # widget.app is a read-only property in Textual, we don't need it for _write_line testing

    # Mock the Log child widget
    log_child = MagicMock()
    widget.query_one = MagicMock(return_value=log_child)

    # Default level is INFO
    widget._min_level = logging.INFO

    # Test DEBUG (should be ignored)
    widget._write_line("Debug message", logging.DEBUG)
    log_child.write_line.assert_not_called()

    # Test INFO (should be written)
    widget._write_line("Info message", logging.INFO)
    log_child.write_line.assert_called_with("Info message")

    # Test ERROR (should be written)
    log_child.reset_mock()
    widget._write_line("Error message", logging.ERROR)
    log_child.write_line.assert_called_with("Error message")

    # Change level to ERROR
    widget._min_level = logging.ERROR

    # Test INFO (should be ignored)
    log_child.reset_mock()
    widget._write_line("Info message", logging.INFO)
    log_child.write_line.assert_not_called()

    # Test ERROR (should be written)
    widget._write_line("Error message", logging.ERROR)
    log_child.write_line.assert_called_with("Error message")

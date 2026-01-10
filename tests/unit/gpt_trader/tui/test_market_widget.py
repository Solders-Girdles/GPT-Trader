"""Tests for MarketWatchWidget."""

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import MarketState
from gpt_trader.tui.widgets.market import MarketWatchWidget


class TestMarketWatchWidget:
    """Tests for MarketWatchWidget."""

    def test_watch_state_handles_none(self):
        """Test that watch_state handles None state gracefully."""
        widget = MarketWatchWidget()
        widget.update_prices = MagicMock()
        widget.watch_state(None)
        widget.update_prices.assert_not_called()

    def test_watch_state_handles_valid_state(self):
        """Test that watch_state processes valid state without errors."""
        widget = MarketWatchWidget()

        # Create a mock state with market data
        mock_state = MagicMock(spec=TuiState)
        mock_state.market_data = MarketState(
            prices={"BTC-USD": Decimal("50000.00"), "ETH-USD": Decimal("3000.00")},
            last_update=1234567890.5,
            price_history={
                "BTC-USD": [Decimal("49900"), Decimal("50000")],
                "ETH-USD": [Decimal("2990"), Decimal("3000")],
            },
        )

        # Mock the update_prices method to avoid widget mounting issues
        widget.update_prices = MagicMock()

        # Should not raise ValueError on f-string formatting
        widget.watch_state(mock_state)

        # Verify update_prices was called with correct data
        widget.update_prices.assert_called_once()
        call_args = widget.update_prices.call_args
        assert call_args[0][0] == mock_state.market_data.prices
        assert call_args[0][1] == mock_state.market_data.last_update
        assert call_args[0][2] == mock_state.market_data.price_history

    def test_watch_state_handles_falsy_last_update(self):
        """Test that watch_state handles falsy (0.0) last_update without errors."""
        widget = MarketWatchWidget()

        # Create a mock state with falsy last_update (0.0 or None via mocking)
        mock_state = MagicMock(spec=TuiState)
        # Mock the attribute to return None to test the `or 0.0` fallback
        mock_state.market_data = MagicMock()
        mock_state.market_data.prices = {"BTC-USD": Decimal("50000.00")}
        mock_state.market_data.last_update = None  # Falsy value that triggers fallback
        mock_state.market_data.price_history = {"BTC-USD": [Decimal("50000")]}

        widget.update_prices = MagicMock()

        # Should not raise ValueError on f-string formatting
        widget.watch_state(mock_state)

        # Verify update_prices was called
        widget.update_prices.assert_called_once()

    def test_watch_state_handles_zero_last_update(self):
        """Test that watch_state handles zero last_update."""
        widget = MarketWatchWidget()

        # Create a mock state with 0.0 last_update
        mock_state = MagicMock(spec=TuiState)
        mock_state.market_data = MarketState(
            prices={"BTC-USD": Decimal("50000.00")},
            last_update=0.0,
            price_history={"BTC-USD": [Decimal("50000")]},
        )

        widget.update_prices = MagicMock()

        # Should not raise ValueError on f-string formatting
        widget.watch_state(mock_state)

        # Verify update_prices was called
        widget.update_prices.assert_called_once()

    def test_watch_state_logs_correct_metrics(self, caplog):
        """Test that watch_state logs expected debug information."""
        import logging

        widget = MarketWatchWidget()

        mock_state = MagicMock(spec=TuiState)
        mock_state.market_data = MarketState(
            prices={"BTC-USD": Decimal("50000.00"), "ETH-USD": Decimal("3000.00")},
            last_update=1234567890.5,
            price_history={},
        )

        widget.update_prices = MagicMock()

        # Capture debug logs
        with caplog.at_level(logging.DEBUG):
            widget.watch_state(mock_state)

        # Verify debug log contains expected information
        debug_logs = [record.message for record in caplog.records if record.levelname == "DEBUG"]
        assert any("[MarketWatchWidget]" in log for log in debug_logs)
        assert any("symbols=2" in log for log in debug_logs)
        assert any("last_update=1234567890.50" in log for log in debug_logs)

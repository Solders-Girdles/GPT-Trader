"""
Tests for TuiState.
"""

from unittest.mock import MagicMock

from gpt_trader.tui.state import TuiState


class TestTuiState:
    def test_initial_state(self):
        state = TuiState()
        assert state.running is False
        assert state.uptime == 0.0
        assert state.market_data.prices == {}
        assert state.position_data.equity == "0.00"

    def test_update_from_bot_status(self):
        state = TuiState()

        status = {
            "market": {"last_prices": {"BTC-USD": "50000.00"}, "last_price_update": 1600000000.0},
            "positions": {
                "equity": "10000.00",
                "total_unrealized_pnl": "500.00",
                "BTC-USD": {"quantity": "0.1"},
            },
            "orders": [
                {
                    "order_id": "1",
                    "symbol": "BTC-USD",
                    "side": "BUY",
                    "quantity": "0.1",
                    "price": "50000.00",
                    "status": "OPEN",
                    "timestamp": 1600000000,
                }
            ],
            "trades": [
                {
                    "symbol": "BTC-USD",
                    "side": "BUY",
                    "quantity": "0.05",
                    "price": "49990.00",
                    "timestamp": 1600000005,
                }
            ],
        }

        mock_runtime = MagicMock()
        mock_runtime.uptime = 120.0

        state.update_from_bot_status(status, mock_runtime)

        assert state.market_data.prices["BTC-USD"] == "50000.00"
        assert state.market_data.last_update == 1600000000.0
        assert state.position_data.equity == "10000.00"
        assert state.position_data.total_unrealized_pnl == "500.00"
        assert state.uptime == 120.0

    def test_reactive_updates(self):
        # This tests that setting properties works as expected for a reactive widget
        state = TuiState()
        state.running = True
        assert state.running is True

"""Tests for TradeRecord."""

from decimal import Decimal

from gpt_trader.features.strategy_dev.monitor.metrics import TradeRecord


class TestTradeRecord:
    def test_open_trade(self):
        """Test open trade."""
        trade = TradeRecord(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )

        assert not trade.is_closed
        assert trade.pnl == 0.0

    def test_closed_trade(self):
        """Test closed trade."""
        trade = TradeRecord(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            exit_price=Decimal("52000"),
            pnl=200.0,
        )

        assert trade.is_closed
        assert trade.is_winner

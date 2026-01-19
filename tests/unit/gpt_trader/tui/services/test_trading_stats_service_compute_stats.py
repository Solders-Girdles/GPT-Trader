"""Tests for TradingStatsService.compute_stats()."""

from decimal import Decimal

from gpt_trader.tui.services.trading_stats_service import TradingStatsService
from gpt_trader.tui.types import Trade


class TestTradingStatsComputation:
    """Tests for compute_stats method."""

    def test_empty_trades_returns_default_stats(self):
        """Test empty trade list returns default TradingStats."""
        service = TradingStatsService()
        stats = service.compute_stats([])

        assert stats.total_trades == 0
        assert stats.win_rate == 0.0
        assert stats.total_pnl == Decimal("0")
        assert stats.window_label == "All Session"

    def test_single_buy_trade_no_match(self):
        """Test single buy trade shows raw trade count (no exit pair)."""
        service = TradingStatsService()

        trade = Trade(
            trade_id="t1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            order_id="o1",
            time="2024-01-15T10:00:00Z",
        )

        stats = service.compute_stats([trade])

        # No matched trades yet, but shows raw trade count
        assert stats.total_trades == 1
        assert stats.winning_trades == 0  # No matched pair
        assert stats.total_pnl == Decimal("0")

    def test_buy_and_sell_creates_matched_trade(self):
        """Test buy followed by sell creates a matched trade pair."""
        service = TradingStatsService()

        buy_trade = Trade(
            trade_id="t1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            order_id="o1",
            time="2024-01-15T10:00:00Z",
        )
        sell_trade = Trade(
            trade_id="t2",
            symbol="BTC-USD",
            side="SELL",
            quantity=Decimal("0.1"),
            price=Decimal("51000"),
            order_id="o2",
            time="2024-01-15T10:30:00Z",
        )

        stats = service.compute_stats([buy_trade, sell_trade])

        # Should have one matched trade
        assert stats.total_trades == 1
        assert stats.winning_trades == 1
        assert stats.losing_trades == 0
        # P&L: (51000 - 50000) * 0.1 = $100
        assert stats.total_pnl == Decimal("100")
        assert stats.win_rate == 1.0

    def test_losing_trade_calculation(self):
        """Test losing trade P&L calculation."""
        service = TradingStatsService()

        buy_trade = Trade(
            trade_id="t1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            order_id="o1",
            time="2024-01-15T10:00:00Z",
        )
        sell_trade = Trade(
            trade_id="t2",
            symbol="BTC-USD",
            side="SELL",
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
            order_id="o2",
            time="2024-01-15T10:30:00Z",
        )

        stats = service.compute_stats([buy_trade, sell_trade])

        assert stats.total_trades == 1
        assert stats.winning_trades == 0
        assert stats.losing_trades == 1
        # P&L: (49000 - 50000) * 0.1 = -$100
        assert stats.total_pnl == Decimal("-100")
        assert stats.win_rate == 0.0

    def test_profit_factor_calculation(self):
        """Test profit factor = gross_profit / gross_loss."""
        service = TradingStatsService()

        trades = [
            # Win: +$100
            Trade(
                trade_id="t1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            Trade(
                trade_id="t2",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("0.1"),
                price=Decimal("51000"),
                order_id="o2",
                time="2024-01-15T10:30:00Z",
            ),
            # Loss: -$50
            Trade(
                trade_id="t3",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.1"),
                price=Decimal("51000"),
                order_id="o3",
                time="2024-01-15T11:00:00Z",
            ),
            Trade(
                trade_id="t4",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("0.1"),
                price=Decimal("50500"),
                order_id="o4",
                time="2024-01-15T11:30:00Z",
            ),
        ]

        stats = service.compute_stats(trades)

        assert stats.total_trades == 2
        assert stats.winning_trades == 1
        assert stats.losing_trades == 1
        assert stats.gross_profit == Decimal("100")
        assert stats.gross_loss == Decimal("50")
        # Profit factor: 100 / 50 = 2.0
        assert stats.profit_factor == 2.0

    def test_profit_factor_infinity_when_no_losses(self):
        """Test profit factor is infinity when there are no losses."""
        service = TradingStatsService()

        trades = [
            Trade(
                trade_id="t1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            Trade(
                trade_id="t2",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("0.1"),
                price=Decimal("51000"),
                order_id="o2",
                time="2024-01-15T10:30:00Z",
            ),
        ]

        stats = service.compute_stats(trades)

        assert stats.profit_factor == float("inf")

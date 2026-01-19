"""Tests for TradingStatsService FIFO trade matching logic."""

from decimal import Decimal

from gpt_trader.tui.services.trading_stats_service import TradingStatsService
from gpt_trader.tui.types import Trade


class TestTradeMatching:
    """Tests for FIFO trade matching logic."""

    def test_fifo_order_matching(self):
        """Test trades are matched in FIFO order."""
        service = TradingStatsService()

        # Two buys at different prices, one sell
        trades = [
            Trade(
                trade_id="t1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),  # First buy at $50k
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            Trade(
                trade_id="t2",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.1"),
                price=Decimal("48000"),  # Second buy at $48k
                order_id="o2",
                time="2024-01-15T10:30:00Z",
            ),
            Trade(
                trade_id="t3",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("0.1"),
                price=Decimal("51000"),  # Sell at $51k - should match first buy
                order_id="o3",
                time="2024-01-15T11:00:00Z",
            ),
        ]

        stats = service.compute_stats(trades)

        # Only one match (first buy + sell)
        assert stats.total_trades == 1
        # P&L from first buy: (51000 - 50000) * 0.1 = $100
        assert stats.total_pnl == Decimal("100")

    def test_partial_fill_matching(self):
        """Test partial fill matching splits quantities correctly."""
        service = TradingStatsService()

        trades = [
            # Buy 0.2 BTC
            Trade(
                trade_id="t1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.2"),
                price=Decimal("50000"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            # Sell 0.1 BTC (partial)
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

        # One matched trade (0.1 BTC)
        assert stats.total_trades == 1
        # P&L: (51000 - 50000) * 0.1 = $100
        assert stats.total_pnl == Decimal("100")

    def test_multiple_symbols_matched_separately(self):
        """Test different symbols are matched separately."""
        service = TradingStatsService()

        trades = [
            # BTC buy
            Trade(
                trade_id="t1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            # ETH buy
            Trade(
                trade_id="t2",
                symbol="ETH-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("3000"),
                order_id="o2",
                time="2024-01-15T10:00:00Z",
            ),
            # BTC sell
            Trade(
                trade_id="t3",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("0.1"),
                price=Decimal("51000"),
                order_id="o3",
                time="2024-01-15T10:30:00Z",
            ),
            # ETH sell
            Trade(
                trade_id="t4",
                symbol="ETH-USD",
                side="SELL",
                quantity=Decimal("1.0"),
                price=Decimal("3100"),
                order_id="o4",
                time="2024-01-15T10:30:00Z",
            ),
        ]

        stats = service.compute_stats(trades)

        # Two matched trades (one per symbol)
        assert stats.total_trades == 2
        assert stats.winning_trades == 2
        # BTC P&L: +$100, ETH P&L: +$100
        assert stats.total_pnl == Decimal("200")

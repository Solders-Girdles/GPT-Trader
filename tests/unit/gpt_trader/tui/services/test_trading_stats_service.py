"""Tests for TradingStatsService."""

from decimal import Decimal

from gpt_trader.tui.services.trading_stats_service import (
    TIME_WINDOWS,
    TradingStatsService,
    clear_trading_stats_service,
    get_trading_stats_service,
)
from gpt_trader.tui.types import Trade, TradingStats


class TestTradingStatsService:
    """Tests for TradingStatsService core functionality."""

    def test_initial_window_is_all_session(self):
        """Test service starts with 'All Session' window."""
        service = TradingStatsService()
        minutes, label = service.current_window
        assert minutes == 0
        assert label == "All Session"

    def test_cycle_window_advances(self):
        """Test cycle_window advances through windows."""
        service = TradingStatsService()

        # Start at All Session (index 0)
        assert service.current_window == (0, "All Session")

        # Cycle to next (5 min)
        minutes, label = service.cycle_window()
        assert minutes == 5
        assert label == "Last 5 min"

        # Cycle to 15 min
        minutes, label = service.cycle_window()
        assert minutes == 15
        assert label == "Last 15 min"

    def test_cycle_window_wraps_around(self):
        """Test cycle_window wraps back to start."""
        service = TradingStatsService()

        # Cycle through all windows
        for _ in range(len(TIME_WINDOWS)):
            service.cycle_window()

        # Should be back at All Session
        assert service.current_window == (0, "All Session")

    def test_reset_window_returns_to_all_session(self):
        """Test reset_window returns to 'All Session'."""
        service = TradingStatsService()

        # Advance to some other window
        service.cycle_window()
        service.cycle_window()

        # Reset
        minutes, label = service.reset_window()
        assert minutes == 0
        assert label == "All Session"
        assert service.current_window == (0, "All Session")


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


class TestTradingStatsSampleSize:
    """Tests for sample size tracking."""

    def test_sample_label_format(self):
        """Test sample_label property format."""
        stats = TradingStats(total_trades=23)
        assert stats.sample_label == "n=23"

    def test_has_sufficient_data_below_threshold(self):
        """Test has_sufficient_data returns False below threshold."""
        stats = TradingStats(total_trades=4)
        assert stats.has_sufficient_data is False

    def test_has_sufficient_data_at_threshold(self):
        """Test has_sufficient_data returns True at threshold."""
        stats = TradingStats(total_trades=5)
        assert stats.has_sufficient_data is True

    def test_has_sufficient_data_above_threshold(self):
        """Test has_sufficient_data returns True above threshold."""
        stats = TradingStats(total_trades=100)
        assert stats.has_sufficient_data is True


class TestTradingStatsServiceSingleton:
    """Tests for singleton pattern."""

    def test_get_returns_same_instance(self):
        """Test get_trading_stats_service returns same instance."""
        clear_trading_stats_service()
        service1 = get_trading_stats_service()
        service2 = get_trading_stats_service()
        assert service1 is service2

    def test_clear_creates_new_instance(self):
        """Test clear_trading_stats_service creates new instance."""
        clear_trading_stats_service()
        service1 = get_trading_stats_service()
        clear_trading_stats_service()
        service2 = get_trading_stats_service()
        assert service1 is not service2


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

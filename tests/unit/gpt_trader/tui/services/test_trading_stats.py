"""Tests for TradingStatsService core functionality, model helpers, and singleton."""

from gpt_trader.tui.services.trading_stats_service import (
    TIME_WINDOWS,
    TradingStatsService,
    clear_trading_stats_service,
    get_trading_stats_service,
)
from gpt_trader.tui.types import TradingStats


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

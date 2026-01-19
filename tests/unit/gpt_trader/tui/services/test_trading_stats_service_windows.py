"""Tests for TradingStatsService window selection behavior."""

from gpt_trader.tui.services.trading_stats_service import TIME_WINDOWS, TradingStatsService


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

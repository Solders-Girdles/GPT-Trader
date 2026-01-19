"""Tests for TradingStatsService singleton helpers."""

from gpt_trader.tui.services.trading_stats_service import (
    clear_trading_stats_service,
    get_trading_stats_service,
)


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

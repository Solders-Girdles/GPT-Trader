"""Tests for TradingStats model helpers."""

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

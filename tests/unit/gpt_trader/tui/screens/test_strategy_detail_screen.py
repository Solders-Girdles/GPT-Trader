"""Tests for StrategyDetailScreen signal detail formatting."""

from gpt_trader.tui.screens.strategy_detail_screen import StrategyDetailScreen
from gpt_trader.tui.types import IndicatorContribution


class TestBuildSignalDetailContent:
    """Tests for _build_signal_detail_content helper."""

    def test_bullish_signal_shows_positive_contribution_and_up_arrow(self):
        """Bullish signal shows positive contribution with up arrow."""
        screen = StrategyDetailScreen()
        contrib = IndicatorContribution(
            name="RSI",
            value=35.20,
            contribution=0.42,
            weight=0.80,
        )

        content = screen._build_signal_detail_content(contrib)

        # Should contain signal name
        assert "RSI" in content
        # Should contain positive contribution with + sign
        assert "+0.42" in content
        # Should contain bullish arrow
        assert "↑" in content

    def test_bearish_signal_shows_negative_contribution_and_down_arrow(self):
        """Bearish signal shows negative contribution with down arrow."""
        screen = StrategyDetailScreen()
        contrib = IndicatorContribution(
            name="MACD",
            value=-0.15,
            contribution=-0.35,
            weight=0.60,
        )

        content = screen._build_signal_detail_content(contrib)

        # Should contain signal name
        assert "MACD" in content
        # Should contain negative contribution (no + sign)
        assert "-0.35" in content
        # Should contain bearish arrow
        assert "↓" in content

    def test_neutral_signal_shows_right_arrow(self):
        """Neutral signal shows near-zero contribution with right arrow."""
        screen = StrategyDetailScreen()
        contrib = IndicatorContribution(
            name="ADX",
            value=25.0,
            contribution=0.005,  # Near zero
            weight=0.50,
        )

        content = screen._build_signal_detail_content(contrib)

        # Should contain signal name
        assert "ADX" in content
        # Should contain neutral arrow
        assert "→" in content

    def test_long_name_truncated_to_ten_characters(self):
        """Long signal names are truncated to 10 characters."""
        screen = StrategyDetailScreen()
        contrib = IndicatorContribution(
            name="VeryLongIndicatorName",
            value=50.0,
            contribution=0.30,
            weight=1.0,
        )

        content = screen._build_signal_detail_content(contrib)

        # Should truncate to first 10 chars
        assert "VeryLongIn" in content
        # Full name should not appear
        assert "VeryLongIndicatorName" not in content

    def test_includes_formatted_value_and_weight(self):
        """Row includes formatted value and weight."""
        screen = StrategyDetailScreen()
        contrib = IndicatorContribution(
            name="RSI",
            value=72.50,
            contribution=0.25,
            weight=0.75,
        )

        content = screen._build_signal_detail_content(contrib)

        # Should contain formatted value
        assert "72.50" in content
        # Should contain weight
        assert "0.75" in content

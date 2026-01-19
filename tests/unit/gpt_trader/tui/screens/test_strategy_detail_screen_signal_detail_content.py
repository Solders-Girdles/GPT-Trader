"""Tests for StrategyDetailScreen signal detail formatting."""

from gpt_trader.tui.screens.strategy_detail_screen import TUNING_HINTS, StrategyDetailScreen
from gpt_trader.tui.types import IndicatorContribution, StrategyParameters


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

        assert "RSI" in content
        assert "+0.42" in content
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

        assert "MACD" in content
        assert "-0.35" in content
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

        assert "ADX" in content
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

        assert "VeryLongIn" in content
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

        assert "72.50" in content
        assert "0.75" in content

    def test_rsi_row_includes_static_hint_when_no_params(self):
        """RSI row includes static tuning hint when no params set."""
        screen = StrategyDetailScreen()
        contrib = IndicatorContribution(
            name="RSI",
            value=35.0,
            contribution=0.30,
            weight=0.80,
        )

        content = screen._build_signal_detail_content(contrib)

        assert "Higher period = slower signals" in content

    def test_rsi_row_includes_live_params_when_set(self):
        """RSI row includes live params when strategy params set."""
        screen = StrategyDetailScreen()
        screen._strategy_params = StrategyParameters(rsi_period=14)
        contrib = IndicatorContribution(
            name="RSI",
            value=35.0,
            contribution=0.30,
            weight=0.80,
        )

        content = screen._build_signal_detail_content(contrib)

        assert "period=14" in content
        assert "Higher period = slower signals" not in content

    def test_unknown_indicator_has_no_hint(self):
        """Unknown indicator row has no hint appended."""
        screen = StrategyDetailScreen()
        contrib = IndicatorContribution(
            name="CustomIndicator",
            value=50.0,
            contribution=0.20,
            weight=1.0,
        )

        content = screen._build_signal_detail_content(contrib)

        for hint in TUNING_HINTS.values():
            assert hint not in content

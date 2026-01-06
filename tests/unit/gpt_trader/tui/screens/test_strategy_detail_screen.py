"""Tests for StrategyDetailScreen signal detail formatting."""

from gpt_trader.tui.screens.strategy_detail_screen import (
    TUNING_HINTS,
    StrategyDetailScreen,
    _get_indicator_hint,
)
from gpt_trader.tui.types import IndicatorContribution, StrategyPerformance


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

    def test_rsi_row_includes_tuning_hint(self):
        """RSI row includes tuning hint text."""
        screen = StrategyDetailScreen()
        contrib = IndicatorContribution(
            name="RSI",
            value=35.0,
            contribution=0.30,
            weight=0.80,
        )

        content = screen._build_signal_detail_content(contrib)

        # Should contain the RSI hint
        assert "Higher period = slower signals" in content

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

        # Should not contain any hint parentheses at end
        # (check that none of the known hints appear)
        for hint in TUNING_HINTS.values():
            assert hint not in content


class TestGetIndicatorHint:
    """Tests for _get_indicator_hint helper."""

    def test_exact_match(self):
        """Exact indicator name returns hint."""
        assert _get_indicator_hint("RSI") == "Higher period = slower signals"
        assert _get_indicator_hint("MACD") == "Wider spread = smoother trend"

    def test_case_insensitive(self):
        """Hint lookup is case-insensitive."""
        assert _get_indicator_hint("rsi") == "Higher period = slower signals"
        assert _get_indicator_hint("Macd") == "Wider spread = smoother trend"

    def test_with_parameters(self):
        """Indicator with parameters extracts first token."""
        assert _get_indicator_hint("RSI(14)") == "Higher period = slower signals"
        assert _get_indicator_hint("MACD_signal") == "Wider spread = smoother trend"
        assert _get_indicator_hint("EMA20") == "Longer EMA = slower trend"

    def test_unknown_returns_none(self):
        """Unknown indicator returns None."""
        assert _get_indicator_hint("CustomIndicator") is None
        assert _get_indicator_hint("XYZ123") is None


class TestBuildBacktestDisplay:
    """Tests for _build_backtest_display helper."""

    def test_build_backtest_display_placeholder(self):
        """None performance returns placeholder values and note."""
        screen = StrategyDetailScreen()

        values, note = screen._build_backtest_display(None)

        assert values == {
            "win_rate": "--",
            "trades": "--",
            "profit_factor": "--",
            "drawdown": "--",
        }
        assert note == "No backtest data available"

    def test_build_backtest_display_values(self):
        """Performance with trades returns formatted values."""
        screen = StrategyDetailScreen()
        performance = StrategyPerformance(
            win_rate=0.652,
            profit_factor=1.85,
            max_drawdown_pct=-3.2,
            total_trades=23,
        )

        values, note = screen._build_backtest_display(performance)

        assert values["win_rate"] == "65.2%"
        assert values["trades"] == "23"
        assert values["profit_factor"] == "1.85"
        assert values["drawdown"] == "-3.2%"
        assert note == ""

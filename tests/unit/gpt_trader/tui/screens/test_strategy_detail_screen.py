"""Tests for StrategyDetailScreen signal detail formatting."""

from gpt_trader.tui.screens.strategy_detail_screen import (
    TUNING_HINTS,
    StrategyDetailScreen,
    _get_indicator_hint,
)
from gpt_trader.tui.types import (
    IndicatorContribution,
    StrategyParameters,
    StrategyPerformance,
)


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

        # Should contain the RSI static hint
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

        # Should contain the live RSI param
        assert "period=14" in content
        # Should NOT contain the static hint
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

        # Should not contain any hint parentheses at end
        # (check that none of the known hints appear)
        for hint in TUNING_HINTS.values():
            assert hint not in content


class TestStrategyParametersFormat:
    """Tests for StrategyParameters.format_indicator_params."""

    def test_rsi_period_formatted(self):
        """RSI with period shows formatted param string."""
        params = StrategyParameters(rsi_period=14)
        assert params.format_indicator_params("RSI") == "period=14"

    def test_ma_shows_all_parts(self):
        """MA with fast, slow, type shows all parts."""
        params = StrategyParameters(ma_fast_period=5, ma_slow_period=20, ma_type="SMA")
        result = params.format_indicator_params("MA")
        assert "fast=5" in result
        assert "slow=20" in result
        assert "type=SMA" in result

    def test_ema_uses_ma_params(self):
        """EMA indicator uses MA params."""
        params = StrategyParameters(ma_fast_period=12, ma_slow_period=26)
        result = params.format_indicator_params("EMA")
        assert "fast=12" in result
        assert "slow=26" in result

    def test_zscore_formatted(self):
        """Z-Score shows lookback and entry threshold."""
        params = StrategyParameters(zscore_lookback=20, zscore_entry_threshold=2.0)
        result = params.format_indicator_params("ZSCORE")
        assert "lookback=20" in result
        assert "entry=2.0" in result

    def test_vwap_deviation_percent(self):
        """VWAP deviation formatted as percentage."""
        params = StrategyParameters(vwap_deviation_threshold=0.01)
        assert params.format_indicator_params("VWAP") == "dev=1.0%"

    def test_spread_tight_bps(self):
        """Spread shows tight threshold in bps."""
        params = StrategyParameters(spread_tight_bps=5.0)
        assert params.format_indicator_params("SPREAD") == "tight=5bps"

    def test_orderbook_formatted(self):
        """Orderbook shows levels and threshold."""
        params = StrategyParameters(orderbook_levels=5, orderbook_imbalance_threshold=0.2)
        result = params.format_indicator_params("ORDERBOOK")
        assert "levels=5" in result
        assert "thresh=20%" in result

    def test_unknown_indicator_returns_none(self):
        """Unknown indicator returns None."""
        params = StrategyParameters(rsi_period=14)
        assert params.format_indicator_params("CUSTOM") is None
        assert params.format_indicator_params("ADX") is None  # Not configured

    def test_no_params_configured_returns_none(self):
        """Indicator with no configured params returns None."""
        params = StrategyParameters()  # All None
        assert params.format_indicator_params("RSI") is None
        assert params.format_indicator_params("MA") is None

    def test_case_insensitive(self):
        """Indicator name lookup is case-insensitive."""
        params = StrategyParameters(rsi_period=14)
        assert params.format_indicator_params("rsi") == "period=14"
        assert params.format_indicator_params("Rsi") == "period=14"


class TestGetIndicatorHint:
    """Tests for _get_indicator_hint helper."""

    def test_exact_match_static_fallback(self):
        """Exact indicator name returns static hint when no params."""
        assert _get_indicator_hint("RSI") == "Higher period = slower signals"
        assert _get_indicator_hint("MACD") == "Wider spread = smoother trend"

    def test_case_insensitive_static(self):
        """Static hint lookup is case-insensitive."""
        assert _get_indicator_hint("rsi") == "Higher period = slower signals"
        assert _get_indicator_hint("Macd") == "Wider spread = smoother trend"

    def test_with_parameters_in_name(self):
        """Indicator with parameters extracts first token for static lookup."""
        assert _get_indicator_hint("RSI(14)") == "Higher period = slower signals"
        assert _get_indicator_hint("MACD_signal") == "Wider spread = smoother trend"
        assert _get_indicator_hint("EMA20") == "Longer EMA = slower trend"

    def test_unknown_returns_none(self):
        """Unknown indicator returns None."""
        assert _get_indicator_hint("CustomIndicator") is None
        assert _get_indicator_hint("XYZ123") is None

    def test_live_params_take_precedence(self):
        """Live params take precedence over static hints."""
        params = StrategyParameters(rsi_period=14)
        assert _get_indicator_hint("RSI", params) == "period=14"

    def test_fallback_to_static_when_param_not_configured(self):
        """Falls back to static when specific param not configured."""
        params = StrategyParameters(rsi_period=14)  # No MACD params
        # MACD not configured, should fall back to static
        assert _get_indicator_hint("MACD", params) == "Wider spread = smoother trend"

    def test_live_params_for_multiple_indicators(self):
        """Multiple indicators with live params."""
        params = StrategyParameters(
            rsi_period=14,
            ma_fast_period=5,
            ma_slow_period=20,
            zscore_lookback=20,
        )
        assert _get_indicator_hint("RSI", params) == "period=14"
        assert "fast=5" in _get_indicator_hint("MA", params)
        assert "lookback=20" in _get_indicator_hint("ZSCORE", params)


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


class TestFormatDelta:
    """Tests for _format_delta helper."""

    def test_positive_delta_shows_green_with_plus(self):
        """Positive delta shows green color with plus sign."""
        screen = StrategyDetailScreen()
        result = screen._format_delta(current=65.5, previous=60.0, suffix="%")
        assert "+5.5%" in result
        assert "green" in result

    def test_negative_delta_shows_red(self):
        """Negative delta shows red color."""
        screen = StrategyDetailScreen()
        result = screen._format_delta(current=55.0, previous=60.0, suffix="%")
        assert "-5.0%" in result
        assert "red" in result

    def test_no_previous_returns_empty(self):
        """No previous value returns empty string."""
        screen = StrategyDetailScreen()
        result = screen._format_delta(current=65.5, previous=None)
        assert result == ""

    def test_tiny_delta_ignored(self):
        """Deltas smaller than 0.01 are ignored."""
        screen = StrategyDetailScreen()
        result = screen._format_delta(current=65.005, previous=65.0)
        assert result == ""

    def test_precision_parameter(self):
        """Precision parameter controls decimal places."""
        screen = StrategyDetailScreen()
        result = screen._format_delta(current=1.85, previous=1.50, precision=2)
        assert "+0.35" in result

    def test_suffix_parameter(self):
        """Suffix parameter appends to formatted value."""
        screen = StrategyDetailScreen()
        result = screen._format_delta(current=10.0, previous=5.0, suffix="%")
        assert "+5.0%" in result


class TestEntryExitBadgeDetailScreen:
    """Tests for _get_entry_exit_badge method on StrategyDetailScreen."""

    def test_buy_action_returns_entry_badge(self):
        """BUY action returns ENTRY badge."""
        screen = StrategyDetailScreen()
        badge = screen._get_entry_exit_badge("BUY")
        assert "ENTRY" in badge
        assert "cyan" in badge

    def test_close_action_returns_exit_badge(self):
        """CLOSE action returns EXIT badge."""
        screen = StrategyDetailScreen()
        badge = screen._get_entry_exit_badge("CLOSE")
        assert "EXIT" in badge
        assert "magenta" in badge

    def test_hold_action_returns_empty(self):
        """HOLD action returns empty string."""
        screen = StrategyDetailScreen()
        badge = screen._get_entry_exit_badge("HOLD")
        assert badge == ""

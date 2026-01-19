"""Tests for StrategyDetailScreen indicator hints and parameter formatting."""

from gpt_trader.tui.screens.strategy_detail_screen import _get_indicator_hint
from gpt_trader.tui.types import StrategyParameters


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

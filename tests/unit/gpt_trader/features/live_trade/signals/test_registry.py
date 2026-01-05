"""Tests for signal registry."""

import pytest

from gpt_trader.features.live_trade.signals.registry import (
    create_signal,
    get_signal_registration,
    list_registered_signals,
)


class TestSignalRegistry:
    """Tests for signal registry functions."""

    def test_list_registered_signals_returns_all_builtin(self):
        """All built-in signals should be registered."""
        signals = list_registered_signals()

        # Core signals
        assert "trend" in signals
        assert "mean_reversion" in signals
        assert "momentum" in signals

        # Microstructure signals
        assert "order_flow" in signals
        assert "orderbook_imbalance" in signals
        assert "spread" in signals
        assert "vwap" in signals

    def test_get_signal_registration_returns_none_for_unknown(self):
        """Unknown signal names should return None."""
        result = get_signal_registration("nonexistent_signal")
        assert result is None

    def test_get_signal_registration_returns_registration_for_known(self):
        """Known signals should return registration info."""
        result = get_signal_registration("trend")
        assert result is not None
        assert result.signal_class is not None
        assert result.config_class is not None

    def test_create_signal_with_default_config(self):
        """Creating signal without parameters uses defaults."""
        signal = create_signal("trend")
        assert signal is not None

    def test_create_signal_with_custom_parameters(self):
        """Creating signal with parameters passes them to config."""
        signal = create_signal("order_flow", {"aggressor_threshold_bullish": 0.7})
        assert signal.config.aggressor_threshold_bullish == 0.7

    def test_create_signal_unknown_raises_value_error(self):
        """Creating unknown signal raises ValueError."""
        with pytest.raises(ValueError, match="Unknown signal"):
            create_signal("nonexistent")

    def test_create_all_registered_signals(self):
        """All registered signals should be creatable."""
        signals = list_registered_signals()
        for name in signals:
            signal = create_signal(name)
            assert signal is not None, f"Failed to create signal: {name}"


class TestSignalRegistrationDetails:
    """Tests for individual signal registrations."""

    @pytest.mark.parametrize(
        "signal_name,expected_desc",
        [
            ("trend", "Trend-following"),
            ("mean_reversion", "Mean reversion"),
            ("momentum", "Momentum"),
            ("order_flow", "Order flow"),
            ("orderbook_imbalance", "Orderbook imbalance"),
            ("spread", "Market quality"),
            ("vwap", "Mean reversion"),
        ],
    )
    def test_signal_has_description(self, signal_name, expected_desc):
        """Each signal should have a meaningful description."""
        reg = get_signal_registration(signal_name)
        assert reg is not None
        assert expected_desc.lower() in reg.description.lower()

    def test_order_flow_signal_config_parameters(self):
        """OrderFlowSignal should accept its specific parameters."""
        params = {
            "aggressor_threshold_bullish": 0.65,
            "aggressor_threshold_bearish": 0.35,
            "min_trades": 15,
            "volume_weight": True,
        }
        signal = create_signal("order_flow", params)
        assert signal.config.aggressor_threshold_bullish == 0.65
        assert signal.config.aggressor_threshold_bearish == 0.35
        assert signal.config.min_trades == 15
        assert signal.config.volume_weight is True

    def test_vwap_signal_config_parameters(self):
        """VWAPSignal should accept its specific parameters."""
        params = {
            "deviation_threshold": 0.015,
            "strong_deviation_threshold": 0.03,
            "min_trades": 30,
        }
        signal = create_signal("vwap", params)
        assert signal.config.deviation_threshold == 0.015
        assert signal.config.strong_deviation_threshold == 0.03
        assert signal.config.min_trades == 30

    def test_spread_signal_config_parameters(self):
        """SpreadSignal should accept its specific parameters."""
        params = {
            "tight_spread_bps": 3.0,
            "normal_spread_bps": 12.0,
            "wide_spread_bps": 25.0,
        }
        signal = create_signal("spread", params)
        assert signal.config.tight_spread_bps == 3.0
        assert signal.config.normal_spread_bps == 12.0
        assert signal.config.wide_spread_bps == 25.0

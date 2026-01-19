"""Tests for ensemble profile configuration dataclasses."""

from __future__ import annotations

from gpt_trader.features.live_trade.strategies.ensemble_profile import (
    CombinerProfileConfig,
    DecisionProfileConfig,
    SignalProfileConfig,
)


class TestSignalProfileConfig:
    """Tests for SignalProfileConfig."""

    def test_from_dict_minimal(self):
        """Create config with just name."""
        config = SignalProfileConfig.from_dict({"name": "trend"})
        assert config.name == "trend"
        assert config.enabled is True
        assert config.parameters == {}

    def test_from_dict_full(self):
        """Create config with all fields."""
        data = {
            "name": "order_flow",
            "enabled": True,
            "parameters": {"aggressor_threshold_bullish": 0.7},
        }
        config = SignalProfileConfig.from_dict(data)
        assert config.name == "order_flow"
        assert config.enabled is True
        assert config.parameters["aggressor_threshold_bullish"] == 0.7

    def test_to_dict_roundtrip(self):
        """to_dict should be reversible."""
        original = SignalProfileConfig(
            name="vwap",
            enabled=True,
            parameters={"min_trades": 25},
        )
        data = original.to_dict()
        restored = SignalProfileConfig.from_dict(data)
        assert restored.name == original.name
        assert restored.enabled == original.enabled
        assert restored.parameters == original.parameters


class TestDecisionProfileConfig:
    """Tests for DecisionProfileConfig."""

    def test_defaults(self):
        """Default values should match EnsembleStrategyConfig."""
        config = DecisionProfileConfig()
        assert config.buy_threshold == 0.2
        assert config.sell_threshold == -0.2
        assert config.close_threshold == 0.1
        assert config.stop_loss_pct == 0.02
        assert config.take_profit_pct == 0.05

    def test_from_dict_custom(self):
        """Custom values should override defaults."""
        data = {
            "buy_threshold": 0.35,
            "sell_threshold": -0.35,
            "stop_loss_pct": 0.015,
        }
        config = DecisionProfileConfig.from_dict(data)
        assert config.buy_threshold == 0.35
        assert config.sell_threshold == -0.35
        assert config.stop_loss_pct == 0.015
        # Defaults for unspecified
        assert config.close_threshold == 0.1


class TestCombinerProfileConfig:
    """Tests for CombinerProfileConfig."""

    def test_defaults(self):
        """Default regime detection thresholds."""
        config = CombinerProfileConfig()
        assert config.adx_period == 14
        assert config.trending_threshold == 25
        assert config.ranging_threshold == 20

    def test_to_regime_config_with_weights(self):
        """Custom weights should be applied."""
        config = CombinerProfileConfig(
            trending_weights={"TREND": 1.2, "ORDER_FLOW": 0.9},
            ranging_weights={"MEAN_REVERSION": 1.0},
        )
        regime_config = config.to_regime_config()

        from gpt_trader.features.live_trade.signals.types import SignalType

        assert regime_config.trending_weights[SignalType.TREND] == 1.2
        assert regime_config.trending_weights[SignalType.ORDER_FLOW] == 0.9
        assert regime_config.ranging_weights[SignalType.MEAN_REVERSION] == 1.0

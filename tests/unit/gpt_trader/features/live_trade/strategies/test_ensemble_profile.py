"""Tests for ensemble profile configuration."""

import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from gpt_trader.features.live_trade.strategies.ensemble import EnsembleStrategy
from gpt_trader.features.live_trade.strategies.ensemble_profile import (
    CombinerProfileConfig,
    DecisionProfileConfig,
    EnsembleProfile,
    SignalProfileConfig,
    get_aggressive_profile,
    get_conservative_profile,
    get_default_profile,
    get_microstructure_profile,
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


class TestEnsembleProfile:
    """Tests for EnsembleProfile."""

    def test_default_profile_is_valid(self):
        """Default profile should pass validation."""
        profile = get_default_profile()
        errors = profile.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_microstructure_profile_is_valid(self):
        """Microstructure profile should pass validation."""
        profile = get_microstructure_profile()
        errors = profile.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_conservative_profile_is_valid(self):
        """Conservative profile should pass validation."""
        profile = get_conservative_profile()
        errors = profile.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_aggressive_profile_is_valid(self):
        """Aggressive profile should pass validation."""
        profile = get_aggressive_profile()
        errors = profile.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_build_signals_creates_instances(self):
        """build_signals should create SignalGenerator instances."""
        profile = get_default_profile()
        signals = profile.build_signals()
        assert len(signals) > 0
        # Check they're valid signal generators
        for signal in signals:
            assert hasattr(signal, "generate")

    def test_build_signals_respects_enabled_flag(self):
        """Disabled signals should not be built."""
        profile = EnsembleProfile(
            name="test",
            signals=[
                SignalProfileConfig(name="trend", enabled=True),
                SignalProfileConfig(name="momentum", enabled=False),
            ],
        )
        signals = profile.build_signals()
        assert len(signals) == 1

    def test_build_combiner(self):
        """build_combiner should create a RegimeAwareCombiner."""
        profile = get_default_profile()
        combiner = profile.build_combiner()
        assert combiner is not None
        assert hasattr(combiner, "combine")

    def test_yaml_roundtrip(self):
        """Profile should survive YAML serialization."""
        profile = get_microstructure_profile()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = Path(f.name)

        try:
            profile.to_yaml(path)
            restored = EnsembleProfile.from_yaml(path)

            assert restored.name == profile.name
            assert restored.version == profile.version
            assert len(restored.signals) == len(profile.signals)
            assert restored.decision.buy_threshold == profile.decision.buy_threshold
        finally:
            path.unlink()

    def test_validation_requires_name(self):
        """Profile without name should fail validation."""
        profile = EnsembleProfile(name="", signals=[SignalProfileConfig(name="trend")])
        errors = profile.validate()
        assert any("name" in e.lower() for e in errors)

    def test_validation_requires_signals(self):
        """Profile without signals should fail validation."""
        profile = EnsembleProfile(name="test", signals=[])
        errors = profile.validate()
        assert any("signal" in e.lower() for e in errors)

    def test_validation_requires_enabled_signals(self):
        """Profile with all disabled signals should fail validation."""
        profile = EnsembleProfile(
            name="test",
            signals=[SignalProfileConfig(name="trend", enabled=False)],
        )
        errors = profile.validate()
        assert any("enabled" in e.lower() for e in errors)


class TestEnsembleStrategyFromProfile:
    """Tests for EnsembleStrategy.from_profile factory methods."""

    def test_from_profile_creates_strategy(self):
        """from_profile should create working EnsembleStrategy."""
        profile = get_default_profile()
        strategy = EnsembleStrategy.from_profile(profile)

        assert strategy is not None
        assert len(strategy.signals) > 0
        assert strategy.combiner is not None
        assert strategy.config.buy_threshold == profile.decision.buy_threshold

    def test_from_profile_name_default(self):
        """from_profile_name('default') should work."""
        strategy = EnsembleStrategy.from_profile_name("default")
        assert strategy is not None

    def test_from_profile_name_microstructure(self):
        """from_profile_name('microstructure') should work."""
        strategy = EnsembleStrategy.from_profile_name("microstructure")
        assert strategy is not None
        # Microstructure should have orderbook/trade flow signals
        signal_names = [type(s).__name__ for s in strategy.signals]
        assert "OrderFlowSignal" in signal_names or "VWAPSignal" in signal_names

    def test_from_profile_name_conservative(self):
        """from_profile_name('conservative') should work."""
        strategy = EnsembleStrategy.from_profile_name("conservative")
        assert strategy is not None
        # Conservative should have higher thresholds
        assert strategy.config.buy_threshold >= 0.3

    def test_from_profile_name_aggressive(self):
        """from_profile_name('aggressive') should work."""
        strategy = EnsembleStrategy.from_profile_name("aggressive")
        assert strategy is not None
        # Aggressive should have lower thresholds
        assert strategy.config.buy_threshold <= 0.2

    def test_from_profile_name_unknown_raises(self):
        """from_profile_name with unknown name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown profile"):
            EnsembleStrategy.from_profile_name("nonexistent_profile")

    def test_from_yaml_file(self):
        """from_yaml should load and create strategy."""
        profile = get_default_profile()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = Path(f.name)

        try:
            profile.to_yaml(path)
            strategy = EnsembleStrategy.from_yaml(path)

            assert strategy is not None
            assert strategy.config.buy_threshold == profile.decision.buy_threshold
        finally:
            path.unlink()


class TestProfileDecisionIntegration:
    """Integration tests for profile-based strategy decisions."""

    def test_strategy_from_profile_can_decide(self):
        """Strategy built from profile should make decisions."""
        strategy = EnsembleStrategy.from_profile_name("default")

        # Create minimal context for decision
        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[Decimal("49000"), Decimal("49500"), Decimal("50000")],
            equity=Decimal("10000"),
            product=None,
        )

        assert decision is not None
        assert decision.action is not None
        assert decision.reason is not None

    def test_different_profiles_different_thresholds(self):
        """Different profiles should have different decision thresholds."""
        conservative = EnsembleStrategy.from_profile_name("conservative")
        aggressive = EnsembleStrategy.from_profile_name("aggressive")

        # Conservative should require stronger signals
        assert conservative.config.buy_threshold > aggressive.config.buy_threshold
        assert conservative.config.stop_loss_pct < aggressive.config.stop_loss_pct

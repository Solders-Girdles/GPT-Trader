"""Tests for ensemble profile configs and utilities."""

from __future__ import annotations

import pytest

from gpt_trader.features.live_trade.signals.types import SignalType
from gpt_trader.features.live_trade.strategies.ensemble_profile import (
    CombinerProfileConfig,
    DecisionProfileConfig,
    EnsembleProfile,
    SignalProfileConfig,
    _parse_signal_type,
    list_ensemble_profiles,
    load_ensemble_profile,
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

        assert regime_config.trending_weights[SignalType.TREND] == 1.2
        assert regime_config.trending_weights[SignalType.ORDER_FLOW] == 0.9
        assert regime_config.ranging_weights[SignalType.MEAN_REVERSION] == 1.0


class TestEnsembleProfileUtilities:
    """Tests for ensemble profile helpers and edge handling."""

    def test_list_profiles_missing_dir_returns_empty(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "gpt_trader.config.path_registry.PROJECT_ROOT",
            tmp_path,
        )

        assert list_ensemble_profiles() == []

    def test_load_profile_missing_raises(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "gpt_trader.config.path_registry.PROJECT_ROOT",
            tmp_path,
        )

        with pytest.raises(FileNotFoundError):
            load_ensemble_profile("missing_profile")

    def test_parse_signal_type_unknown_returns_none(self) -> None:
        assert _parse_signal_type("unknown") is None

    def test_build_signals_skips_unknown(self) -> None:
        profile = EnsembleProfile(
            name="edge",
            signals=[
                SignalProfileConfig(name="trend"),
                SignalProfileConfig(name="unknown"),
            ],
        )

        signals = profile.build_signals()

        assert len(signals) == 1

    def test_combiner_config_ignores_invalid_weight_keys(self) -> None:
        config = CombinerProfileConfig(
            trending_weights={"TREND": 0.7, "BOGUS": 0.1},
            ranging_weights={"MEAN_REVERSION": 0.9, "MISSING": 0.2},
        )

        regime_config = config.to_regime_config()

        assert regime_config.trending_weights[SignalType.TREND] == 0.7
        assert regime_config.ranging_weights[SignalType.MEAN_REVERSION] == 0.9


def test_validate_flags_invalid_thresholds() -> None:
    decision = DecisionProfileConfig(
        buy_threshold=0.0,
        sell_threshold=0.0,
        stop_loss_pct=1.0,
    )
    profile = EnsembleProfile(
        name="invalid",
        signals=[SignalProfileConfig(name="trend")],
        decision=decision,
    )

    errors = profile.validate()

    assert "buy_threshold must be positive" in errors
    assert "sell_threshold must be negative" in errors
    assert "stop_loss_pct must be between 0 and 1" in errors

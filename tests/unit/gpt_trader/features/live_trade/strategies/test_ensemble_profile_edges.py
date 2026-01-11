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


def test_list_profiles_missing_dir_returns_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "gpt_trader.config.path_registry.PROJECT_ROOT",
        tmp_path,
    )

    assert list_ensemble_profiles() == []


def test_load_profile_missing_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "gpt_trader.config.path_registry.PROJECT_ROOT",
        tmp_path,
    )

    with pytest.raises(FileNotFoundError):
        load_ensemble_profile("missing_profile")


def test_parse_signal_type_unknown_returns_none() -> None:
    assert _parse_signal_type("unknown") is None


def test_build_signals_skips_unknown() -> None:
    profile = EnsembleProfile(
        name="edge",
        signals=[
            SignalProfileConfig(name="trend"),
            SignalProfileConfig(name="unknown"),
        ],
    )

    signals = profile.build_signals()

    assert len(signals) == 1


def test_combiner_config_ignores_invalid_weight_keys() -> None:
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

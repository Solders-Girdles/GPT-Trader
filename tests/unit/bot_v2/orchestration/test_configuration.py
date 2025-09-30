"""Unit tests for orchestration configuration primitives."""

import pytest

from bot_v2.orchestration.configuration import (
    BotConfig,
    ConfigManager,
    ConfigValidationError,
    Profile,
)


def test_dev_profile_defaults(monkeypatch):
    monkeypatch.delenv("SPOT_FORCE_LIVE", raising=False)
    config = BotConfig.from_profile("dev")
    assert config.profile is Profile.DEV
    assert config.mock_broker is True
    assert config.mock_fills is True
    assert config.dry_run is True


def test_spot_profile_normalizes_perp_symbols(monkeypatch):
    monkeypatch.delenv("SPOT_FORCE_LIVE", raising=False)
    config = BotConfig.from_profile("spot", symbols=["btc-perp", "ETH-USD"])
    assert config.profile is Profile.SPOT
    # BTC-PERP should be converted to BTC-USD while preserving existing spot entries
    assert list(config.symbols) == ["BTC-USD", "ETH-USD"]
    assert config.enable_shorts is False
    assert config.max_leverage == 1


def test_canary_overrides_honor_locks(monkeypatch):
    monkeypatch.delenv("SPOT_FORCE_LIVE", raising=False)
    config = BotConfig.from_profile(
        "canary", max_leverage=5, time_in_force="GTC", reduce_only_mode=False
    )
    # Canary profile must remain IOC + reduce-only regardless of overrides
    assert config.profile is Profile.CANARY
    assert config.max_leverage == 1
    assert config.reduce_only_mode is True
    assert config.time_in_force == "IOC"


def test_config_manager_detects_invalid_symbols(monkeypatch):
    monkeypatch.delenv("SPOT_FORCE_LIVE", raising=False)
    # Empty/whitespace symbols are filtered out and replaced with defaults
    manager = ConfigManager(profile=Profile.DEV, overrides={"symbols": ["", "   "]})
    config = manager.build()
    # Should fall back to default spot symbols (BTC-USD, ETH-USD)
    assert len(config.symbols) > 0
    assert all(s.strip() for s in config.symbols)


def test_has_changes_reflects_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SPOT_FORCE_LIVE", raising=False)
    manager = ConfigManager(profile="dev")
    assert manager.has_changes() is False
    monkeypatch.setenv("COINBASE_DEFAULT_QUOTE", "USDT")
    assert manager.has_changes() is True

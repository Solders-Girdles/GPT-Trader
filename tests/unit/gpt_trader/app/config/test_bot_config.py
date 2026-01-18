"""Tests for BotConfig feature flag canonicalization and precedence."""

from __future__ import annotations

import os
import warnings
from unittest.mock import patch

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.app.config.bot_config import MeanReversionConfig
from gpt_trader.features.live_trade.strategies.perps_baseline import PerpsStrategyConfig


class TestEnableShortsCanonical:
    """Test enable_shorts derivation from strategy config with sync warning."""

    def test_active_enable_shorts_from_baseline_strategy(self) -> None:
        """Baseline strategy type derives enable_shorts from strategy config."""
        config = BotConfig(
            strategy=PerpsStrategyConfig(enable_shorts=True),
            strategy_type="baseline",
        )
        # Reset warning state for clean test
        BotConfig._enable_shorts_sync_warned = False

        with pytest.warns(UserWarning, match="differs from strategy config"):
            assert config.active_enable_shorts is True

    def test_active_enable_shorts_from_mean_reversion(self) -> None:
        """Mean reversion strategy type derives enable_shorts from mean_reversion config."""
        config = BotConfig(
            mean_reversion=MeanReversionConfig(enable_shorts=False),
            strategy_type="mean_reversion",
        )
        BotConfig._enable_shorts_sync_warned = False

        assert config.active_enable_shorts is False

    def test_sync_warning_on_mismatch(self) -> None:
        """Warns when BotConfig.enable_shorts differs from strategy config."""
        config = BotConfig(
            enable_shorts=True,  # Top-level says True
            strategy=PerpsStrategyConfig(enable_shorts=False),  # Strategy says False
            strategy_type="baseline",
        )
        BotConfig._enable_shorts_sync_warned = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = config.active_enable_shorts

        assert result is False  # Strategy config is canonical
        assert len(w) == 1
        assert "differs from strategy config" in str(w[0].message)

    def test_no_warning_when_synced(self) -> None:
        """No warning when BotConfig.enable_shorts matches strategy config."""
        config = BotConfig(
            enable_shorts=True,
            strategy=PerpsStrategyConfig(enable_shorts=True),
            strategy_type="baseline",
        )
        BotConfig._enable_shorts_sync_warned = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = config.active_enable_shorts

        assert len(w) == 0

    def test_warning_only_once_per_process(self) -> None:
        """Sync warning fires only once per process."""
        config = BotConfig(
            enable_shorts=True,
            strategy=PerpsStrategyConfig(enable_shorts=False),
            strategy_type="baseline",
        )
        BotConfig._enable_shorts_sync_warned = False

        # First access triggers warning
        with warnings.catch_warnings(record=True) as w1:
            warnings.simplefilter("always")
            _ = config.active_enable_shorts
        assert len(w1) == 1

        # Second access does not
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            _ = config.active_enable_shorts
        assert len(w2) == 0


class TestMockBrokerEnvParsing:
    """Test MOCK_BROKER env parsing."""

    def test_mock_broker_env_true(self) -> None:
        """MOCK_BROKER=1 enables mock broker."""
        with patch.dict(os.environ, {"MOCK_BROKER": "1"}, clear=True):
            result = BotConfig.from_env().mock_broker
        assert result is True

    def test_mock_broker_env_default_false(self) -> None:
        """Defaults to False when MOCK_BROKER is unset."""
        with patch.dict(os.environ, {}, clear=True):
            result = BotConfig.from_env().mock_broker
        assert result is False


class TestReduceOnlyModeEnvParsing:
    """Test reduce_only_mode env variable parsing."""

    def test_risk_prefixed_enabled(self) -> None:
        """RISK_REDUCE_ONLY_MODE enables reduce-only mode."""
        env = {
            "RISK_REDUCE_ONLY_MODE": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch.dict(os.environ, {"BROKER": "coinbase"}):
                config = BotConfig.from_env()
        assert config.reduce_only_mode is True

    def test_default_false(self) -> None:
        """Defaults to False when no env vars set."""
        with patch.dict(os.environ, {"BROKER": "coinbase"}, clear=True):
            config = BotConfig.from_env()
        assert config.reduce_only_mode is False


class TestDerivativesEnvParsing:
    """Test derivatives/perps environment flag parsing."""

    def test_derivatives_enabled_prefers_intx_perps_flag(self) -> None:
        """COINBASE_ENABLE_INTX_PERPS overrides legacy COINBASE_ENABLE_DERIVATIVES."""
        env = {
            "COINBASE_ENABLE_INTX_PERPS": "1",
            "COINBASE_ENABLE_DERIVATIVES": "0",
        }
        with patch.dict(os.environ, env, clear=True):
            config = BotConfig.from_env()
        assert config.derivatives_enabled is True

    def test_derivatives_enabled_intx_perps_can_disable_legacy(self) -> None:
        """COINBASE_ENABLE_INTX_PERPS=0 disables derivatives even if legacy is 1."""
        env = {
            "COINBASE_ENABLE_INTX_PERPS": "0",
            "COINBASE_ENABLE_DERIVATIVES": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            config = BotConfig.from_env()
        assert config.derivatives_enabled is False

    def test_derivatives_enabled_falls_back_to_legacy(self) -> None:
        """When COINBASE_ENABLE_INTX_PERPS is unset, legacy flag still works."""
        env = {
            "COINBASE_ENABLE_DERIVATIVES": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.warns(DeprecationWarning, match="COINBASE_ENABLE_DERIVATIVES"):
                config = BotConfig.from_env()
        assert config.derivatives_enabled is True

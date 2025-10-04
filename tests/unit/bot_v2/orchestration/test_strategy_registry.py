"""Tests for StrategyRegistry - strategy initialization and retrieval.

This module tests the StrategyRegistry's ability to:
- Initialize per-symbol strategies for SPOT profile
- Initialize shared strategy for PERPS profile
- Retrieve strategies with lazy creation
- Handle configuration overrides and validation
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.strategy_registry import StrategyRegistry


@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    return Mock()


@pytest.fixture
def spot_config(mock_risk_manager):
    """Create SPOT profile configuration."""
    config = Mock()
    config.profile = Profile.SPOT
    config.symbols = ["BTC-USD", "ETH-USD"]
    config.short_ma = 20
    config.long_ma = 50
    config.trailing_stop_pct = 0.02
    config.perps_position_fraction = None
    config.derivatives_enabled = False
    return config


@pytest.fixture
def perps_config(mock_risk_manager):
    """Create PERPS profile configuration (using PROD profile)."""
    config = Mock()
    config.profile = Profile.PROD  # Non-SPOT profiles use PERPS/derivatives mode
    config.symbols = ["BTC-PERP"]
    config.short_ma = 20
    config.long_ma = 50
    config.target_leverage = 2
    config.trailing_stop_pct = 0.02
    config.enable_shorts = True
    config.perps_position_fraction = None
    config.derivatives_enabled = True
    return config


@pytest.fixture
def mock_spot_profiles():
    """Create mock spot profile service."""
    return Mock()


class TestInitializeSpotProfile:
    """Test strategy initialization for SPOT profile."""

    def test_creates_per_symbol_strategies(
        self, spot_config, mock_risk_manager, mock_spot_profiles
    ):
        """Creates separate strategy for each symbol."""
        mock_spot_profiles.load.return_value = {
            "BTC-USD": {},
            "ETH-USD": {},
        }

        registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
        registry.initialize()

        assert "BTC-USD" in registry.symbol_strategies
        assert "ETH-USD" in registry.symbol_strategies
        assert len(registry.symbol_strategies) == 2

    def test_uses_custom_windows_from_spot_profiles(
        self, spot_config, mock_risk_manager, mock_spot_profiles
    ):
        """Uses custom short/long windows from spot profile rules."""
        mock_spot_profiles.load.return_value = {
            "BTC-USD": {"short_window": 10, "long_window": 30},
            "ETH-USD": {"short_window": 15, "long_window": 40},
        }

        registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
        registry.initialize()

        btc_strategy = registry.symbol_strategies["BTC-USD"]
        eth_strategy = registry.symbol_strategies["ETH-USD"]

        assert btc_strategy.config.short_ma_period == 10
        assert btc_strategy.config.long_ma_period == 30
        assert eth_strategy.config.short_ma_period == 15
        assert eth_strategy.config.long_ma_period == 40

    def test_uses_default_windows_when_not_in_profile(
        self, spot_config, mock_risk_manager, mock_spot_profiles
    ):
        """Falls back to config defaults when windows not in spot profile."""
        mock_spot_profiles.load.return_value = {"BTC-USD": {}}

        registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
        registry.initialize()

        strategy = registry.symbol_strategies["BTC-USD"]
        assert strategy.config.short_ma_period == 20  # From spot_config.short_ma
        assert strategy.config.long_ma_period == 50  # From spot_config.long_ma

    def test_applies_position_fraction_override_from_profile(
        self, spot_config, mock_risk_manager, mock_spot_profiles
    ):
        """Applies position_fraction override from spot profile."""
        mock_spot_profiles.load.return_value = {
            "BTC-USD": {"position_fraction": 0.25},
        }

        registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
        registry.initialize()

        strategy = registry.symbol_strategies["BTC-USD"]
        assert strategy.config.position_fraction == 0.25

    def test_applies_global_position_fraction_when_profile_has_none(
        self, spot_config, mock_risk_manager, mock_spot_profiles
    ):
        """Uses global position_fraction when profile has none."""
        spot_config.perps_position_fraction = 0.3
        mock_spot_profiles.load.return_value = {"BTC-USD": {}}

        registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
        registry.initialize()

        strategy = registry.symbol_strategies["BTC-USD"]
        assert strategy.config.position_fraction == 0.3

    def test_handles_invalid_position_fraction_gracefully(
        self, spot_config, mock_risk_manager, mock_spot_profiles, caplog
    ):
        """Handles invalid position_fraction without crashing."""
        mock_spot_profiles.load.return_value = {
            "BTC-USD": {"position_fraction": "invalid"},
        }

        with caplog.at_level("WARNING"):
            registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
            registry.initialize()

        assert "BTC-USD" in registry.symbol_strategies
        assert "Invalid position_fraction" in caplog.text

    def test_spot_strategies_have_correct_defaults(
        self, spot_config, mock_risk_manager, mock_spot_profiles
    ):
        """SPOT strategies have correct default settings."""
        mock_spot_profiles.load.return_value = {"BTC-USD": {}}

        registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
        registry.initialize()

        strategy = registry.symbol_strategies["BTC-USD"]
        assert strategy.config.target_leverage == 1
        assert strategy.config.enable_shorts is False
        assert strategy.config.trailing_stop_pct == 0.02


class TestInitializePerpsProfile:
    """Test strategy initialization for PERPS profile."""

    def test_creates_shared_strategy(self, perps_config, mock_risk_manager):
        """Creates single shared strategy for PERPS profile."""
        registry = StrategyRegistry(perps_config, mock_risk_manager)
        registry.initialize()

        assert registry.default_strategy is not None
        assert len(registry.symbol_strategies) == 0  # No per-symbol strategies

    def test_uses_config_ma_periods(self, perps_config, mock_risk_manager):
        """Uses MA periods from config."""
        registry = StrategyRegistry(perps_config, mock_risk_manager)
        registry.initialize()

        strategy = registry.default_strategy
        assert strategy.config.short_ma_period == 20
        assert strategy.config.long_ma_period == 50

    def test_applies_derivatives_settings(self, perps_config, mock_risk_manager):
        """Applies derivatives settings when enabled."""
        registry = StrategyRegistry(perps_config, mock_risk_manager)
        registry.initialize()

        strategy = registry.default_strategy
        assert strategy.config.target_leverage == 2
        assert strategy.config.enable_shorts is True

    def test_disables_derivatives_features_when_disabled(self, perps_config, mock_risk_manager):
        """Disables derivatives features when derivatives_enabled is False."""
        perps_config.derivatives_enabled = False

        registry = StrategyRegistry(perps_config, mock_risk_manager)
        registry.initialize()

        strategy = registry.default_strategy
        assert strategy.config.target_leverage == 1
        assert strategy.config.enable_shorts is False

    def test_applies_position_fraction_override(self, perps_config, mock_risk_manager):
        """Applies position_fraction override."""
        perps_config.perps_position_fraction = 0.4

        registry = StrategyRegistry(perps_config, mock_risk_manager)
        registry.initialize()

        strategy = registry.default_strategy
        assert strategy.config.position_fraction == 0.4

    def test_handles_invalid_position_fraction(self, perps_config, mock_risk_manager, caplog):
        """Handles invalid PERPS_POSITION_FRACTION gracefully."""
        perps_config.perps_position_fraction = "invalid"

        with caplog.at_level("WARNING"):
            registry = StrategyRegistry(perps_config, mock_risk_manager)
            registry.initialize()

        assert registry.default_strategy is not None
        assert "Invalid PERPS_POSITION_FRACTION" in caplog.text


class TestGetStrategy:
    """Test strategy retrieval."""

    def test_returns_existing_spot_symbol_strategy(
        self, spot_config, mock_risk_manager, mock_spot_profiles
    ):
        """Returns existing strategy for SPOT symbol."""
        mock_spot_profiles.load.return_value = {"BTC-USD": {}}

        registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
        registry.initialize()

        strategy1 = registry.get_strategy("BTC-USD")
        strategy2 = registry.get_strategy("BTC-USD")

        assert strategy1 is strategy2  # Same instance

    def test_creates_default_strategy_for_missing_spot_symbol(
        self, spot_config, mock_risk_manager, mock_spot_profiles
    ):
        """Creates default strategy for missing SPOT symbol (lazy creation)."""
        mock_spot_profiles.load.return_value = {"BTC-USD": {}}

        registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
        registry.initialize()

        # Request strategy for symbol not in initial config
        strategy = registry.get_strategy("SOL-USD")

        assert strategy is not None
        assert "SOL-USD" in registry.symbol_strategies

    def test_caches_lazily_created_strategy(
        self, spot_config, mock_risk_manager, mock_spot_profiles
    ):
        """Caches lazily created strategy for future requests."""
        mock_spot_profiles.load.return_value = {"BTC-USD": {}}

        registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
        registry.initialize()

        strategy1 = registry.get_strategy("SOL-USD")
        strategy2 = registry.get_strategy("SOL-USD")

        assert strategy1 is strategy2  # Same cached instance

    def test_returns_shared_strategy_for_perps_profile(self, perps_config, mock_risk_manager):
        """Returns shared strategy for PERPS profile."""
        registry = StrategyRegistry(perps_config, mock_risk_manager)
        registry.initialize()

        strategy1 = registry.get_strategy("BTC-PERP")
        strategy2 = registry.get_strategy("ETH-PERP")

        assert strategy1 is strategy2  # Same shared instance
        assert strategy1 is registry.default_strategy


class TestProperties:
    """Test registry properties."""

    def test_default_strategy_property(self, perps_config, mock_risk_manager):
        """default_strategy property returns initialized strategy."""
        registry = StrategyRegistry(perps_config, mock_risk_manager)
        registry.initialize()

        assert registry.default_strategy is not None

    def test_default_strategy_raises_when_not_initialized(self, perps_config, mock_risk_manager):
        """default_strategy raises error when not initialized."""
        registry = StrategyRegistry(perps_config, mock_risk_manager)

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = registry.default_strategy

    def test_symbol_strategies_property(self, spot_config, mock_risk_manager, mock_spot_profiles):
        """symbol_strategies property returns strategy dict."""
        mock_spot_profiles.load.return_value = {"BTC-USD": {}}

        registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
        registry.initialize()

        strategies = registry.symbol_strategies
        assert isinstance(strategies, dict)
        assert "BTC-USD" in strategies


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_symbols_list_spot_profile(
        self, spot_config, mock_risk_manager, mock_spot_profiles
    ):
        """Handles empty symbols list for SPOT profile."""
        spot_config.symbols = []
        mock_spot_profiles.load.return_value = {}

        registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
        registry.initialize()

        assert len(registry.symbol_strategies) == 0

    def test_none_symbols_list_spot_profile(
        self, spot_config, mock_risk_manager, mock_spot_profiles
    ):
        """Handles None symbols list for SPOT profile."""
        spot_config.symbols = None
        mock_spot_profiles.load.return_value = {}

        registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
        registry.initialize()

        assert len(registry.symbol_strategies) == 0

    def test_multiple_symbols_initialization(
        self, spot_config, mock_risk_manager, mock_spot_profiles
    ):
        """Initializes multiple symbols correctly."""
        spot_config.symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
        mock_spot_profiles.load.return_value = {
            "BTC-USD": {"short_window": 10},
            "ETH-USD": {"short_window": 15},
            "SOL-USD": {"short_window": 20},
            "AVAX-USD": {"short_window": 25},
        }

        registry = StrategyRegistry(spot_config, mock_risk_manager, mock_spot_profiles)
        registry.initialize()

        assert len(registry.symbol_strategies) == 4
        assert all(
            symbol in registry.symbol_strategies
            for symbol in ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
        )

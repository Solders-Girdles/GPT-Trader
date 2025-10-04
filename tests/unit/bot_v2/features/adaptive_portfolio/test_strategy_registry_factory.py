"""Tests for strategy registry factory - config-driven handler instantiation.

This module tests the build_strategy_registry factory function that creates
handler registries from tier configuration.

Critical behaviors tested:
- Building registry from tier config strategies list
- Instantiating correct handler types
- Wiring dependencies properly
- Handling ML enhanced wrapper requirement
- Error cases (unknown strategies, missing dependencies)
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from bot_v2.features.adaptive_portfolio.strategy_handlers import (
    MeanReversionStrategyHandler,
    MLEnhancedStrategyHandler,
    MomentumStrategyHandler,
    TrendFollowingStrategyHandler,
)
from bot_v2.features.adaptive_portfolio.strategy_registry_factory import (
    StrategyRegistryError,
    build_strategy_registry,
    get_supported_strategies,
)
from bot_v2.features.adaptive_portfolio.types import (
    PositionConstraints,
    RiskProfile,
    TierConfig,
    TradingRules,
)


@pytest.fixture
def mock_data_provider() -> Mock:
    """Create mock data provider."""
    return Mock()


@pytest.fixture
def mock_position_size_calculator() -> Mock:
    """Create mock position size calculator."""
    return Mock()


class TestGetSupportedStrategies:
    """Test supported strategies enumeration."""

    def test_returns_all_supported_strategy_ids(self) -> None:
        """Returns list of all supported strategy IDs."""
        strategies = get_supported_strategies()

        assert "momentum" in strategies
        assert "mean_reversion" in strategies
        assert "trend_following" in strategies
        assert "ml_enhanced" in strategies

    def test_returns_sorted_list(self) -> None:
        """Returns sorted list for consistent output."""
        strategies = get_supported_strategies()

        # All except ml_enhanced should be sorted
        basic_strategies = [s for s in strategies if s != "ml_enhanced"]
        assert basic_strategies == sorted(basic_strategies)


class TestBuildStrategyRegistry:
    """Test strategy registry building from tier config."""

    def test_builds_registry_for_single_strategy(
        self, mock_data_provider: Mock, mock_position_size_calculator: Mock
    ) -> None:
        """Builds registry with single strategy."""
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        registry = build_strategy_registry(
            tier_config, mock_data_provider, mock_position_size_calculator
        )

        assert len(registry) == 1
        assert "momentum" in registry
        assert isinstance(registry["momentum"], MomentumStrategyHandler)

    def test_builds_registry_for_multiple_strategies(
        self, mock_data_provider: Mock, mock_position_size_calculator: Mock
    ) -> None:
        """Builds registry with multiple strategies."""
        tier_config = TierConfig(
            name="medium",
            range=(10000, 100000),
            positions=PositionConstraints(3, 10, 6, 500.0),
            min_position_size=500.0,
            strategies=["momentum", "mean_reversion", "trend_following"],
            risk=RiskProfile(1.5, 8.0, 4.0, 40.0),
            trading=TradingRules(10, "margin", 1, True),
        )

        registry = build_strategy_registry(
            tier_config, mock_data_provider, mock_position_size_calculator
        )

        assert len(registry) == 3
        assert isinstance(registry["momentum"], MomentumStrategyHandler)
        assert isinstance(registry["mean_reversion"], MeanReversionStrategyHandler)
        assert isinstance(registry["trend_following"], TrendFollowingStrategyHandler)

    def test_wires_dependencies_to_handlers(
        self, mock_data_provider: Mock, mock_position_size_calculator: Mock
    ) -> None:
        """Wires data provider and position calculator to handlers."""
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        registry = build_strategy_registry(
            tier_config, mock_data_provider, mock_position_size_calculator
        )

        handler = registry["momentum"]
        assert handler.data_provider is mock_data_provider
        assert handler.position_size_calculator is mock_position_size_calculator

    def test_ml_enhanced_wraps_momentum_handler(
        self, mock_data_provider: Mock, mock_position_size_calculator: Mock
    ) -> None:
        """ML enhanced handler wraps momentum handler."""
        tier_config = TierConfig(
            name="large",
            range=(100000, float("inf")),
            positions=PositionConstraints(10, 30, 20, 1000.0),
            min_position_size=1000.0,
            strategies=["momentum", "ml_enhanced"],
            risk=RiskProfile(1.0, 5.0, 3.0, 25.0),
            trading=TradingRules(20, "margin", 0, False),
        )

        registry = build_strategy_registry(
            tier_config, mock_data_provider, mock_position_size_calculator
        )

        assert len(registry) == 2
        assert isinstance(registry["ml_enhanced"], MLEnhancedStrategyHandler)
        assert isinstance(registry["ml_enhanced"].momentum_handler, MomentumStrategyHandler)
        # ML enhanced should wrap the same momentum instance from registry
        assert registry["ml_enhanced"].momentum_handler is registry["momentum"]

    def test_raises_error_for_unknown_strategy(
        self, mock_data_provider: Mock, mock_position_size_calculator: Mock
    ) -> None:
        """Raises StrategyRegistryError for unknown strategy ID."""
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum", "unknown_strategy"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        with pytest.raises(StrategyRegistryError) as exc_info:
            build_strategy_registry(tier_config, mock_data_provider, mock_position_size_calculator)

        assert "unknown_strategy" in str(exc_info.value)
        assert "micro" in str(exc_info.value)

    def test_raises_error_for_ml_enhanced_without_momentum(
        self, mock_data_provider: Mock, mock_position_size_calculator: Mock
    ) -> None:
        """Raises StrategyRegistryError when ml_enhanced requested without momentum."""
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["mean_reversion", "ml_enhanced"],  # ml_enhanced needs momentum
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        with pytest.raises(StrategyRegistryError) as exc_info:
            build_strategy_registry(tier_config, mock_data_provider, mock_position_size_calculator)

        assert "ml_enhanced" in str(exc_info.value)
        assert "momentum" in str(exc_info.value)

    def test_handles_empty_strategies_list(
        self, mock_data_provider: Mock, mock_position_size_calculator: Mock
    ) -> None:
        """Returns empty registry for tier with no strategies."""
        tier_config = TierConfig(
            name="micro",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=[],  # No strategies
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        registry = build_strategy_registry(
            tier_config, mock_data_provider, mock_position_size_calculator
        )

        assert registry == {}

    def test_builds_all_four_strategies(
        self, mock_data_provider: Mock, mock_position_size_calculator: Mock
    ) -> None:
        """Builds registry with all supported strategies."""
        tier_config = TierConfig(
            name="large",
            range=(100000, float("inf")),
            positions=PositionConstraints(10, 30, 20, 1000.0),
            min_position_size=1000.0,
            strategies=["momentum", "mean_reversion", "trend_following", "ml_enhanced"],
            risk=RiskProfile(1.0, 5.0, 3.0, 25.0),
            trading=TradingRules(20, "margin", 0, False),
        )

        registry = build_strategy_registry(
            tier_config, mock_data_provider, mock_position_size_calculator
        )

        assert len(registry) == 4
        assert all(
            strategy in registry
            for strategy in ["momentum", "mean_reversion", "trend_following", "ml_enhanced"]
        )

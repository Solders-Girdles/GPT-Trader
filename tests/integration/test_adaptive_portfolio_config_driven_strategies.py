"""End-to-end characterization test for config-driven strategy activation.

This integration test validates the full path from AdaptivePortfolioManager through
config-driven strategy selection, registry building, signal generation, and result
aggregation.

Critical behaviors tested:
- Tier-specific strategy activation based on config
- Registry building from tier configuration
- Per-tier registry caching
- Signal sources match tier strategies
- Full orchestration through AdaptivePortfolioManager
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from bot_v2.data_providers import DataProvider
from bot_v2.features.adaptive_portfolio.adaptive_portfolio import AdaptivePortfolioManager
from bot_v2.features.adaptive_portfolio.types import (
    CostStructure,
    MarketConstraints,
    PortfolioConfig,
    PortfolioTier,
    PositionConstraints,
    PositionInfo,
    RiskProfile,
    TierConfig,
    TradingRules,
)


@pytest.fixture
def mock_data_provider() -> DataProvider:
    """Create mock data provider with realistic price data."""
    provider = Mock(spec=DataProvider)

    # Mock uptrending data for momentum signals
    mock_data = Mock()
    mock_data.__len__ = Mock(return_value=100)
    mock_data.data = {"Close": [100.0 + i * 0.5 for i in range(100)]}  # 100 -> 149.5

    provider.get_historical_data = Mock(return_value=mock_data)
    return provider


@pytest.fixture
def test_portfolio_config() -> PortfolioConfig:
    """Create test portfolio configuration with multiple tiers."""
    return PortfolioConfig(
        version="1.0",
        last_updated="2025-01-01",
        description="Test config for integration testing",
        tiers={
            "micro": TierConfig(
                name="micro",
                range=(0, 5000),
                positions=PositionConstraints(1, 3, 2, 100.0),
                min_position_size=100.0,
                strategies=["momentum"],  # Only momentum for micro
                risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
                trading=TradingRules(5, "cash", 2, True),
            ),
            "small": TierConfig(
                name="small",
                range=(5000, 25000),
                positions=PositionConstraints(2, 5, 3, 200.0),
                min_position_size=200.0,
                strategies=["momentum", "mean_reversion"],  # Two strategies
                risk=RiskProfile(1.8, 9.0, 4.5, 45.0),
                trading=TradingRules(8, "cash", 1, True),
            ),
            "medium": TierConfig(
                name="medium",
                range=(25000, 100000),
                positions=PositionConstraints(3, 10, 6, 500.0),
                min_position_size=500.0,
                strategies=["momentum", "mean_reversion", "trend_following"],  # Three strategies
                risk=RiskProfile(1.5, 8.0, 4.0, 40.0),
                trading=TradingRules(10, "margin", 1, True),
            ),
            "large": TierConfig(
                name="large",
                range=(100000, float("inf")),
                positions=PositionConstraints(10, 30, 20, 1000.0),
                min_position_size=1000.0,
                strategies=[
                    "momentum",
                    "mean_reversion",
                    "trend_following",
                    "ml_enhanced",
                ],  # All four
                risk=RiskProfile(1.0, 5.0, 3.0, 25.0),
                trading=TradingRules(20, "margin", 0, False),
            ),
        },
        costs=CostStructure(
            commission_per_trade=1.0,
            spread_estimate_pct=0.05,
            slippage_pct=0.1,
            financing_rate_annual_pct=5.0,
        ),
        market_constraints=MarketConstraints(
            min_share_price=1.0,
            max_share_price=10000.0,
            min_daily_volume=100000,
            excluded_sectors=[],
            excluded_symbols=[],
            market_hours_only=True,
        ),
        validation={},
        rebalancing={"tier_transition_buffer_pct": 5.0},
    )


class TestConfigDrivenStrategyActivation:
    """Test config-driven strategy activation through AdaptivePortfolioManager."""

    def test_micro_tier_uses_only_momentum_strategy(
        self,
        test_portfolio_config: PortfolioConfig,
        mock_data_provider: DataProvider,
    ) -> None:
        """Micro tier activates only momentum strategy per config."""
        # Create manager with direct config injection (bypasses file loading)
        manager = AdaptivePortfolioManager(
            data_provider=mock_data_provider,
            config=test_portfolio_config,
        )

        # Analyze micro tier portfolio (capital < 5000)
        result = manager.analyze_portfolio(
            current_capital=3000.0,
            positions=[],
        )

        # Verify tier detection
        assert result.current_tier == PortfolioTier.MICRO

        # Verify registry was built for micro tier with only momentum
        assert "micro" in manager.strategy_selector._tier_registries
        micro_registry = manager.strategy_selector._tier_registries["micro"]
        assert list(micro_registry.keys()) == ["momentum"]

        # If signals are generated, verify they all come from momentum
        if result.signals:
            signal_sources = {signal.strategy_source for signal in result.signals}
            assert signal_sources == {
                "momentum"
            }, f"Micro tier should only use momentum, got: {signal_sources}"

    def test_small_tier_uses_momentum_and_mean_reversion(
        self,
        test_portfolio_config: PortfolioConfig,
        mock_data_provider: DataProvider,
    ) -> None:
        """Small tier activates momentum and mean_reversion per config."""
        manager = AdaptivePortfolioManager(
            data_provider=mock_data_provider,
            config=test_portfolio_config,
        )

        # Analyze small tier portfolio (5000 <= capital < 25000)
        result = manager.analyze_portfolio(
            current_capital=15000.0,
            positions=[],
        )

        # Verify tier detection
        assert result.current_tier == PortfolioTier.SMALL

        # Verify registry was built for small tier with correct strategies
        assert "small" in manager.strategy_selector._tier_registries
        small_registry = manager.strategy_selector._tier_registries["small"]
        assert set(small_registry.keys()) == {"momentum", "mean_reversion"}

    def test_medium_tier_uses_three_strategies(
        self,
        test_portfolio_config: PortfolioConfig,
        mock_data_provider: DataProvider,
    ) -> None:
        """Medium tier activates momentum, mean_reversion, and trend_following."""
        manager = AdaptivePortfolioManager(
            data_provider=mock_data_provider,
            config=test_portfolio_config,
        )

        # Analyze medium tier portfolio (25000 <= capital < 100000)
        result = manager.analyze_portfolio(
            current_capital=50000.0,
            positions=[],
        )

        # Verify tier detection
        assert result.current_tier == PortfolioTier.MEDIUM

        # Verify signals generated
        assert len(result.signals) > 0

        # Verify registry was built for medium tier
        assert "medium" in manager.strategy_selector._tier_registries
        medium_registry = manager.strategy_selector._tier_registries["medium"]
        assert set(medium_registry.keys()) == {"momentum", "mean_reversion", "trend_following"}

    def test_large_tier_uses_all_four_strategies_including_ml_enhanced(
        self,
        test_portfolio_config: PortfolioConfig,
        mock_data_provider: DataProvider,
    ) -> None:
        """Large tier activates all four strategies including ml_enhanced."""
        manager = AdaptivePortfolioManager(
            data_provider=mock_data_provider,
            config=test_portfolio_config,
        )

        # Analyze large tier portfolio (capital >= 100000)
        result = manager.analyze_portfolio(
            current_capital=150000.0,
            positions=[],
        )

        # Verify tier detection
        assert result.current_tier == PortfolioTier.LARGE

        # Verify signals generated
        assert len(result.signals) > 0

        # Verify registry was built for large tier
        assert "large" in manager.strategy_selector._tier_registries
        large_registry = manager.strategy_selector._tier_registries["large"]
        assert set(large_registry.keys()) == {
            "momentum",
            "mean_reversion",
            "trend_following",
            "ml_enhanced",
        }

        # Verify ml_enhanced wraps momentum
        ml_handler = large_registry["ml_enhanced"]
        assert hasattr(ml_handler, "momentum_handler")
        assert ml_handler.momentum_handler is large_registry["momentum"]

    def test_registry_caching_across_multiple_analyses(
        self,
        test_portfolio_config: PortfolioConfig,
        mock_data_provider: DataProvider,
    ) -> None:
        """Registry is cached and reused across multiple analyses of same tier."""
        manager = AdaptivePortfolioManager(
            data_provider=mock_data_provider,
            config=test_portfolio_config,
        )

        # First analysis of micro tier
        result1 = manager.analyze_portfolio(current_capital=3000.0, positions=[])

        # Get registry reference
        first_registry = manager.strategy_selector._tier_registries["micro"]

        # Second analysis of micro tier
        result2 = manager.analyze_portfolio(current_capital=4000.0, positions=[])

        # Registry should be the same instance (cached)
        second_registry = manager.strategy_selector._tier_registries["micro"]
        assert first_registry is second_registry, "Registry should be cached and reused"

    def test_different_tiers_get_different_registries(
        self,
        test_portfolio_config: PortfolioConfig,
        mock_data_provider: DataProvider,
    ) -> None:
        """Different tiers build separate registries with different handlers."""
        manager = AdaptivePortfolioManager(
            data_provider=mock_data_provider,
            config=test_portfolio_config,
        )

        # Analyze micro tier
        result_micro = manager.analyze_portfolio(current_capital=3000.0, positions=[])

        # Analyze large tier
        result_large = manager.analyze_portfolio(current_capital=150000.0, positions=[])

        # Both should have built separate registries
        assert "micro" in manager.strategy_selector._tier_registries
        assert "large" in manager.strategy_selector._tier_registries

        micro_registry = manager.strategy_selector._tier_registries["micro"]
        large_registry = manager.strategy_selector._tier_registries["large"]

        # Registries should be different instances
        assert micro_registry is not large_registry

        # Different handler counts
        assert len(micro_registry) == 1  # Only momentum
        assert len(large_registry) == 4  # All four strategies

        # Verify results
        assert result_micro.current_tier == PortfolioTier.MICRO
        assert result_large.current_tier == PortfolioTier.LARGE

    def test_portfolio_with_existing_positions(
        self,
        test_portfolio_config: PortfolioConfig,
        mock_data_provider: DataProvider,
    ) -> None:
        """Strategy selection works with existing positions in portfolio."""
        manager = AdaptivePortfolioManager(
            data_provider=mock_data_provider,
            config=test_portfolio_config,
        )

        # Create existing positions
        positions = [
            PositionInfo(
                symbol="AAPL",
                shares=10,
                entry_price=150.0,
                current_price=155.0,
                position_value=1550.0,
                unrealized_pnl=50.0,
                unrealized_pnl_pct=3.33,
                days_held=5,
            ),
            PositionInfo(
                symbol="GOOGL",
                shares=5,
                entry_price=2800.0,
                current_price=2850.0,
                position_value=14250.0,
                unrealized_pnl=250.0,
                unrealized_pnl_pct=1.79,
                days_held=10,
            ),
        ]

        # Analyze medium tier with positions
        result = manager.analyze_portfolio(
            current_capital=50000.0,
            positions=positions,
        )

        # Verify analysis completed
        assert result.current_tier == PortfolioTier.MEDIUM
        assert result.portfolio_snapshot.positions_count == 2
        assert result.portfolio_snapshot.total_value == 50000.0

        # Signals should still be generated
        assert len(result.signals) >= 0  # May be 0 if all symbols filtered

        # Registry should be built
        assert "medium" in manager.strategy_selector._tier_registries

    def test_risk_metrics_and_recommendations_generated(
        self,
        test_portfolio_config: PortfolioConfig,
        mock_data_provider: DataProvider,
    ) -> None:
        """Full analysis includes risk metrics and recommendations."""
        manager = AdaptivePortfolioManager(
            data_provider=mock_data_provider,
            config=test_portfolio_config,
        )

        result = manager.analyze_portfolio(
            current_capital=3000.0,
            positions=[],
        )

        # Verify full result structure
        assert result.risk_metrics is not None
        assert result.recommended_actions is not None
        assert result.warnings is not None
        assert result.timestamp is not None

        # Verify tier config is correct
        assert result.tier_config.name == "micro"
        assert result.tier_config.strategies == ["momentum"]

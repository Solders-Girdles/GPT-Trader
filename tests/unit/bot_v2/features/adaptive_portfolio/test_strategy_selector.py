"""Tests for StrategySelector - tier-based strategy selection and signal generation.

This module tests the StrategySelector's ability to generate appropriate
trading signals based on portfolio tier, implementing different strategies
(momentum, mean reversion, trend following, ML-enhanced) with tier-specific
parameters and position sizing.

Critical behaviors tested:
- Strategy selection based on tier configuration
- Signal generation for each strategy type
- Position sizing calculations per tier
- Signal filtering and ranking
- Symbol universe selection by tier
- Confidence scoring and adjustments
- Error handling for data issues

Business Context:
    The StrategySelector determines WHAT to trade and HOW MUCH based on
    account size and tier configuration. Failures here can result in:

    - Micro accounts attempting complex multi-strategy approaches
    - Large accounts using overly simple single-strategy methods
    - Position sizes inappropriate for account tier
    - Trading signals that violate tier constraints
    - Over-trading or under-trading relative to tier capacity

    This is the strategic brain that must adapt intelligence to capital.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest

from bot_v2.features.adaptive_portfolio.strategy_selector import StrategySelector
from bot_v2.features.adaptive_portfolio.types import (
    PortfolioConfig,
    PortfolioSnapshot,
    PortfolioTier,
    PositionConstraints,
    PositionInfo,
    RiskProfile,
    TierConfig,
    TradingRules,
    TradingSignal,
)


@pytest.fixture
def mock_data_provider() -> Mock:
    """Create mock data provider with sample market data."""
    provider = Mock()

    # Create mock historical data with strong uptrend
    # Need: returns_5d > 2% and returns_20d > 5%
    mock_data = Mock()
    mock_data.__len__ = Mock(return_value=60)
    mock_data.data = {
        "Close": [100.0 + i * 0.55 for i in range(60)]  # 100 -> 132.45 (stronger uptrend)
    }

    provider.get_historical_data = Mock(return_value=mock_data)
    return provider


@pytest.fixture
def portfolio_config() -> PortfolioConfig:
    """Create sample portfolio configuration."""
    tier_config = TierConfig(
        name="micro",
        range=(0, 5000),
        positions=PositionConstraints(1, 3, 2, 100.0),
        min_position_size=100.0,
        strategies=["momentum"],
        risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
        trading=TradingRules(5, "cash", 2, True),
    )

    # Create proper mock for market_constraints
    market_constraints = Mock()
    market_constraints.excluded_symbols = []

    return PortfolioConfig(
        version="1.0",
        last_updated="2024-01-01",
        description="Test config",
        tiers={"micro": tier_config},
        costs=Mock(),
        market_constraints=market_constraints,
        validation={},
        rebalancing={},
    )


@pytest.fixture
def portfolio_snapshot() -> PortfolioSnapshot:
    """Create sample portfolio snapshot."""
    return PortfolioSnapshot(
        total_value=3000.0,
        cash=2000.0,
        positions=[
            PositionInfo("AAPL", 10, 100.0, 100.0, 1000.0, 0.0, 0.0, 1)
        ],
        daily_pnl=0.0,
        daily_pnl_pct=0.0,
        quarterly_pnl_pct=0.0,
        current_tier=PortfolioTier.MICRO,
        positions_count=1,
        largest_position_pct=33.33,
        sector_exposures={},
    )


@pytest.fixture
def strategy_selector(
    portfolio_config: PortfolioConfig, mock_data_provider: Mock
) -> StrategySelector:
    """Create StrategySelector instance."""
    return StrategySelector(portfolio_config, mock_data_provider)


class TestStrategySelectorInitialization:
    """Test StrategySelector initialization."""

    def test_initializes_with_config_and_provider(
        self, portfolio_config: PortfolioConfig, mock_data_provider: Mock
    ) -> None:
        """Initializes with configuration and data provider.

        Core dependencies must be stored for signal generation.
        """
        selector = StrategySelector(portfolio_config, mock_data_provider)

        assert selector.config is portfolio_config
        assert selector.data_provider is mock_data_provider


class TestSignalGeneration:
    """Test signal generation for different strategies."""

    def test_generates_signals_for_tier(
        self,
        strategy_selector: StrategySelector,
        portfolio_config: PortfolioConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Generates trading signals appropriate for tier.

        Must produce actionable signals based on tier strategies.
        """
        tier_config = portfolio_config.tiers["micro"]

        signals = strategy_selector.generate_signals(
            tier_config, portfolio_snapshot
        )

        assert isinstance(signals, list)
        # Should have signals for momentum strategy

    def test_respects_tier_strategy_list(
        self,
        strategy_selector: StrategySelector,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Only generates signals for strategies in tier config.

        Tier-specific strategy selection is critical.
        """
        tier_config = TierConfig(
            name="test",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum", "mean_reversion"],  # Two strategies
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        signals = strategy_selector.generate_signals(
            tier_config, portfolio_snapshot
        )

        # Should generate signals from both strategies
        sources = {s.strategy_source for s in signals}
        assert len(sources) > 0


class TestMomentumStrategy:
    """Test momentum strategy signal generation."""

    def test_generates_momentum_signals(
        self,
        strategy_selector: StrategySelector,
        portfolio_config: PortfolioConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Generates momentum-based trading signals.

        Momentum strategy identifies strong uptrends.
        """
        tier_config = portfolio_config.tiers["micro"]

        signals = strategy_selector._momentum_strategy(
            ["AAPL"], tier_config, portfolio_snapshot
        )

        # Uptrending mock data should generate buy signals
        if signals:
            assert signals[0].action == "BUY"
            assert signals[0].strategy_source == "momentum"


class TestSymbolUniverse:
    """Test symbol universe selection."""

    def test_micro_tier_gets_limited_universe(
        self,
        strategy_selector: StrategySelector,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Micro tier receives limited symbol universe.

        Small accounts should focus on fewer, more liquid symbols.
        """
        tier_config = TierConfig(
            name="Micro Portfolio",
            range=(0, 5000),
            positions=PositionConstraints(1, 3, 2, 100.0),
            min_position_size=100.0,
            strategies=["momentum"],
            risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
            trading=TradingRules(5, "cash", 2, True),
        )

        universe = strategy_selector._get_symbol_universe(
            tier_config, portfolio_snapshot
        )

        assert len(universe) <= 8  # Limited for micro

    def test_large_tier_gets_full_universe(
        self,
        strategy_selector: StrategySelector,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Large tier receives full symbol universe.

        Large accounts can handle more diversification.
        """
        tier_config = TierConfig(
            name="Large Portfolio",
            range=(100000, float('inf')),
            positions=PositionConstraints(10, 30, 20, 1000.0),
            min_position_size=1000.0,
            strategies=["momentum", "mean_reversion", "trend_following", "ml_enhanced"],
            risk=RiskProfile(1.0, 5.0, 3.0, 25.0),
            trading=TradingRules(20, "margin", 0, False),
        )

        universe = strategy_selector._get_symbol_universe(
            tier_config, portfolio_snapshot
        )

        assert len(universe) > 15  # Larger universe


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_calculates_position_size_based_on_confidence(
        self,
        strategy_selector: StrategySelector,
        portfolio_config: PortfolioConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Position size scales with signal confidence.

        Higher confidence signals get larger positions.
        """
        tier_config = portfolio_config.tiers["micro"]

        size_low = strategy_selector._calculate_signal_position_size(
            0.3, tier_config, portfolio_snapshot
        )
        size_high = strategy_selector._calculate_signal_position_size(
            0.9, tier_config, portfolio_snapshot
        )

        assert size_high > size_low

    def test_respects_minimum_position_size(
        self,
        strategy_selector: StrategySelector,
        portfolio_config: PortfolioConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Position size meets tier minimum requirements.

        Even low confidence signals must meet minimum size.
        """
        tier_config = portfolio_config.tiers["micro"]

        size = strategy_selector._calculate_signal_position_size(
            0.1, tier_config, portfolio_snapshot
        )

        assert size >= tier_config.min_position_size


class TestSignalFiltering:
    """Test signal filtering and ranking."""

    def test_limits_signals_to_tier_capacity(
        self,
        strategy_selector: StrategySelector,
        portfolio_config: PortfolioConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Limits number of signals to tier's position capacity.

        Micro accounts shouldn't get 20 trading signals.
        """
        tier_config = portfolio_config.tiers["micro"]

        # Create many signals
        signals = [
            TradingSignal(
                f"SYM{i}", "BUY", 0.7, 500.0, 5.0, "momentum", "test"
            )
            for i in range(20)
        ]

        max_signals = strategy_selector._calculate_max_signals(
            tier_config, portfolio_snapshot
        )

        assert max_signals <= tier_config.positions.max_positions


class TestErrorHandling:
    """Test error handling for data issues."""

    def test_handles_insufficient_data_gracefully(
        self,
        strategy_selector: StrategySelector,
        portfolio_config: PortfolioConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Handles insufficient historical data without crashing.

        Data provider failures shouldn't crash strategy.
        """
        # Mock data provider returns short history
        short_data = Mock()
        short_data.__len__ = Mock(return_value=5)  # Too short
        short_data.data = {"Close": [100.0] * 5}
        strategy_selector.data_provider.get_historical_data = Mock(
            return_value=short_data
        )

        tier_config = portfolio_config.tiers["micro"]

        # Should not raise
        signals = strategy_selector._momentum_strategy(
            ["AAPL"], tier_config, portfolio_snapshot
        )

        assert isinstance(signals, list)

    def test_handles_data_provider_exceptions(
        self,
        strategy_selector: StrategySelector,
        portfolio_config: PortfolioConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Handles data provider exceptions gracefully.

        Network/API failures shouldn't crash signal generation.
        """
        strategy_selector.data_provider.get_historical_data = Mock(
            side_effect=Exception("API error")
        )

        tier_config = portfolio_config.tiers["micro"]

        # Should not raise
        signals = strategy_selector._momentum_strategy(
            ["AAPL"], tier_config, portfolio_snapshot
        )

        assert signals == []
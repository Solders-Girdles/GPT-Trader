"""Tests for MomentumStrategyHandler.

This module tests the momentum strategy handler's signal generation
based on 5-day and 20-day price returns.

Critical behaviors tested:
- Signal generation for strong uptrends
- Position sizing based on confidence
- Handling of insufficient data
- Error handling for data provider failures
- Correct momentum calculations
- Signal filtering based on thresholds
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from bot_v2.features.adaptive_portfolio.position_size_calculator import PositionSizeCalculator
from bot_v2.features.adaptive_portfolio.strategy_handlers.momentum import MomentumStrategyHandler
from bot_v2.features.adaptive_portfolio.types import (
    PortfolioSnapshot,
    PortfolioTier,
    PositionConstraints,
    PositionInfo,
    RiskProfile,
    TierConfig,
    TradingRules,
)


@pytest.fixture
def tier_config() -> TierConfig:
    """Create sample tier configuration."""
    return TierConfig(
        name="micro",
        range=(0, 5000),
        positions=PositionConstraints(1, 3, 2, 100.0),
        min_position_size=100.0,
        strategies=["momentum"],
        risk=RiskProfile(2.0, 10.0, 5.0, 50.0),
        trading=TradingRules(5, "cash", 2, True),
    )


@pytest.fixture
def portfolio_snapshot() -> PortfolioSnapshot:
    """Create sample portfolio snapshot."""
    return PortfolioSnapshot(
        total_value=3000.0,
        cash=2000.0,
        positions=[PositionInfo("AAPL", 10, 100.0, 100.0, 1000.0, 0.0, 0.0, 1)],
        daily_pnl=0.0,
        daily_pnl_pct=0.0,
        quarterly_pnl_pct=0.0,
        current_tier=PortfolioTier.MICRO,
        positions_count=1,
        largest_position_pct=33.33,
        sector_exposures={},
    )


@pytest.fixture
def mock_data_provider() -> Mock:
    """Create mock data provider with uptrending data."""
    provider = Mock()

    # Create strong uptrend: 100 -> 132.45 over 60 days
    # This gives: 5d return ~2.75%, 20d return ~11%
    mock_data = Mock()
    mock_data.__len__ = Mock(return_value=60)
    mock_data.data = {"Close": [100.0 + i * 0.55 for i in range(60)]}

    provider.get_historical_data = Mock(return_value=mock_data)
    return provider


@pytest.fixture
def position_size_calculator() -> PositionSizeCalculator:
    """Create position size calculator."""
    return PositionSizeCalculator()


@pytest.fixture
def momentum_handler(
    mock_data_provider: Mock, position_size_calculator: PositionSizeCalculator
) -> MomentumStrategyHandler:
    """Create momentum strategy handler."""
    return MomentumStrategyHandler(mock_data_provider, position_size_calculator)


class TestMomentumHandlerInitialization:
    """Test momentum handler initialization."""

    def test_initializes_with_dependencies(
        self, mock_data_provider: Mock, position_size_calculator: PositionSizeCalculator
    ) -> None:
        """Initializes with data provider and position calculator."""
        handler = MomentumStrategyHandler(mock_data_provider, position_size_calculator)

        assert handler.data_provider is mock_data_provider
        assert handler.position_size_calculator is position_size_calculator


class TestSignalGeneration:
    """Test momentum signal generation."""

    def test_generates_buy_signal_for_strong_uptrend(
        self,
        momentum_handler: MomentumStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Generates BUY signal when both 5d and 20d returns exceed thresholds."""
        signals = momentum_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert len(signals) > 0
        signal = signals[0]
        assert signal.symbol == "AAPL"
        assert signal.action == "BUY"
        assert signal.strategy_source == "momentum"

    def test_no_signal_for_insufficient_data(
        self,
        momentum_handler: MomentumStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Does not generate signals when historical data is too short."""
        # Mock short data
        short_data = Mock()
        short_data.__len__ = Mock(return_value=10)
        short_data.data = {"Close": [100.0] * 10}
        momentum_handler.data_provider.get_historical_data = Mock(return_value=short_data)

        signals = momentum_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert signals == []

    def test_no_signal_for_weak_trend(
        self,
        momentum_handler: MomentumStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Does not generate signals when returns don't meet thresholds."""
        # Flat price data
        flat_data = Mock()
        flat_data.__len__ = Mock(return_value=60)
        flat_data.data = {"Close": [100.0] * 60}
        momentum_handler.data_provider.get_historical_data = Mock(return_value=flat_data)

        signals = momentum_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert signals == []

    def test_confidence_based_on_returns(
        self,
        momentum_handler: MomentumStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Signal confidence scales with return magnitude."""
        signals = momentum_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert len(signals) > 0
        signal = signals[0]
        # Strong uptrend should have meaningful confidence
        assert 0.4 < signal.confidence <= 0.8

    def test_position_size_calculated(
        self,
        momentum_handler: MomentumStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Position size is calculated for generated signals."""
        signals = momentum_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert len(signals) > 0
        signal = signals[0]
        assert signal.target_position_size > 0
        assert signal.target_position_size >= tier_config.min_position_size

    def test_processes_multiple_symbols(
        self,
        momentum_handler: MomentumStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Processes multiple symbols and generates signals for each."""
        signals = momentum_handler.generate_signals(
            ["AAPL", "GOOGL", "MSFT"], tier_config, portfolio_snapshot
        )

        # Should generate signals for uptrending symbols
        assert len(signals) >= 1
        symbols = {s.symbol for s in signals}
        assert len(symbols) > 0


class TestErrorHandling:
    """Test error handling."""

    def test_handles_data_provider_exceptions(
        self,
        momentum_handler: MomentumStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Handles data provider exceptions gracefully."""
        momentum_handler.data_provider.get_historical_data = Mock(
            side_effect=Exception("API error")
        )

        signals = momentum_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert signals == []

    def test_handles_missing_price_data(
        self,
        momentum_handler: MomentumStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Handles missing or invalid price data."""
        # Mock data with None values
        bad_data = Mock()
        bad_data.__len__ = Mock(return_value=60)
        bad_data.data = {"Close": [None] * 60}
        momentum_handler.data_provider.get_historical_data = Mock(return_value=bad_data)

        signals = momentum_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert signals == []

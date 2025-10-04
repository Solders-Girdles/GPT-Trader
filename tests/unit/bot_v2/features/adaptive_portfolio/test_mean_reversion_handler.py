"""Tests for MeanReversionStrategyHandler.

This module tests the mean reversion strategy handler's signal generation
based on z-score deviations from moving averages.

Critical behaviors tested:
- Signal generation for oversold conditions
- Z-score calculations
- Position sizing based on confidence
- Handling of insufficient data
- Error handling for data provider failures
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from bot_v2.features.adaptive_portfolio.position_size_calculator import PositionSizeCalculator
from bot_v2.features.adaptive_portfolio.strategy_handlers.mean_reversion import (
    MeanReversionStrategyHandler,
)
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
        strategies=["mean_reversion"],
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
def mock_data_provider_oversold() -> Mock:
    """Create mock data provider with oversold condition."""
    provider = Mock()

    # Create oversold data: prices around 100, recent drop to 85
    # This creates a z-score < -1.5
    prices = [100.0] * 40 + [95.0, 92.0, 90.0, 88.0, 85.0]
    mock_data = Mock()
    mock_data.__len__ = Mock(return_value=len(prices))
    mock_data.data = {"Close": prices}

    provider.get_historical_data = Mock(return_value=mock_data)
    return provider


@pytest.fixture
def position_size_calculator() -> PositionSizeCalculator:
    """Create position size calculator."""
    return PositionSizeCalculator()


@pytest.fixture
def mean_reversion_handler(
    mock_data_provider_oversold: Mock, position_size_calculator: PositionSizeCalculator
) -> MeanReversionStrategyHandler:
    """Create mean reversion strategy handler."""
    return MeanReversionStrategyHandler(mock_data_provider_oversold, position_size_calculator)


class TestMeanReversionHandlerInitialization:
    """Test mean reversion handler initialization."""

    def test_initializes_with_dependencies(
        self, mock_data_provider_oversold: Mock, position_size_calculator: PositionSizeCalculator
    ) -> None:
        """Initializes with data provider and position calculator."""
        handler = MeanReversionStrategyHandler(
            mock_data_provider_oversold, position_size_calculator
        )

        assert handler.data_provider is mock_data_provider_oversold
        assert handler.position_size_calculator is position_size_calculator


class TestSignalGeneration:
    """Test mean reversion signal generation."""

    def test_generates_buy_signal_for_oversold(
        self,
        mean_reversion_handler: MeanReversionStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Generates BUY signal when z-score indicates oversold condition."""
        signals = mean_reversion_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert len(signals) > 0
        signal = signals[0]
        assert signal.symbol == "AAPL"
        assert signal.action == "BUY"
        assert signal.strategy_source == "mean_reversion"
        assert "oversold" in signal.reasoning.lower()

    def test_no_signal_for_insufficient_data(
        self,
        mean_reversion_handler: MeanReversionStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Does not generate signals when historical data is too short."""
        short_data = Mock()
        short_data.__len__ = Mock(return_value=10)
        short_data.data = {"Close": [100.0] * 10}
        mean_reversion_handler.data_provider.get_historical_data = Mock(return_value=short_data)

        signals = mean_reversion_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert signals == []

    def test_no_signal_for_neutral_zscore(
        self,
        mean_reversion_handler: MeanReversionStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Does not generate signals when z-score is neutral."""
        # Flat price data
        flat_data = Mock()
        flat_data.__len__ = Mock(return_value=60)
        flat_data.data = {"Close": [100.0] * 60}
        mean_reversion_handler.data_provider.get_historical_data = Mock(return_value=flat_data)

        signals = mean_reversion_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert signals == []

    def test_confidence_based_on_zscore_magnitude(
        self,
        mean_reversion_handler: MeanReversionStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Signal confidence scales with z-score magnitude."""
        signals = mean_reversion_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert len(signals) > 0
        signal = signals[0]
        # Oversold condition should have meaningful confidence
        assert 0.3 < signal.confidence <= 0.8

    def test_position_size_calculated(
        self,
        mean_reversion_handler: MeanReversionStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Position size is calculated for generated signals."""
        signals = mean_reversion_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert len(signals) > 0
        signal = signals[0]
        assert signal.target_position_size > 0
        assert signal.target_position_size >= tier_config.min_position_size

    def test_processes_multiple_symbols(
        self,
        mean_reversion_handler: MeanReversionStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Processes multiple symbols and generates signals."""
        signals = mean_reversion_handler.generate_signals(
            ["AAPL", "GOOGL", "MSFT"], tier_config, portfolio_snapshot
        )

        # Should generate signals for oversold symbols
        assert len(signals) >= 1


class TestErrorHandling:
    """Test error handling."""

    def test_handles_data_provider_exceptions(
        self,
        mean_reversion_handler: MeanReversionStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Handles data provider exceptions gracefully."""
        mean_reversion_handler.data_provider.get_historical_data = Mock(
            side_effect=Exception("API error")
        )

        signals = mean_reversion_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert signals == []

    def test_handles_missing_price_data(
        self,
        mean_reversion_handler: MeanReversionStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Handles missing or invalid price data."""
        bad_data = Mock()
        bad_data.__len__ = Mock(return_value=60)
        bad_data.data = {"Close": [None] * 60}
        mean_reversion_handler.data_provider.get_historical_data = Mock(return_value=bad_data)

        signals = mean_reversion_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert signals == []

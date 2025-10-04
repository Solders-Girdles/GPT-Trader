"""Tests for TrendFollowingStrategyHandler.

This module tests the trend following strategy handler's signal generation
based on moving average alignment (10/30/50 MA).

Critical behaviors tested:
- Signal generation for MA alignment
- Trend strength calculations
- Position sizing based on confidence
- Handling of insufficient data
- Error handling for data provider failures
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from bot_v2.features.adaptive_portfolio.position_size_calculator import PositionSizeCalculator
from bot_v2.features.adaptive_portfolio.strategy_handlers.trend_following import (
    TrendFollowingStrategyHandler,
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
        strategies=["trend_following"],
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
def mock_data_provider_uptrend() -> Mock:
    """Create mock data provider with strong uptrend."""
    provider = Mock()

    # Create strong uptrend: prices consistently rising
    # Will have sma_10 > sma_30 > sma_50 alignment
    mock_data = Mock()
    mock_data.__len__ = Mock(return_value=100)
    mock_data.data = {"Close": [100.0 + i * 0.5 for i in range(100)]}  # 100 -> 149.5

    provider.get_historical_data = Mock(return_value=mock_data)
    return provider


@pytest.fixture
def position_size_calculator() -> PositionSizeCalculator:
    """Create position size calculator."""
    return PositionSizeCalculator()


@pytest.fixture
def trend_following_handler(
    mock_data_provider_uptrend: Mock, position_size_calculator: PositionSizeCalculator
) -> TrendFollowingStrategyHandler:
    """Create trend following strategy handler."""
    return TrendFollowingStrategyHandler(mock_data_provider_uptrend, position_size_calculator)


class TestTrendFollowingHandlerInitialization:
    """Test trend following handler initialization."""

    def test_initializes_with_dependencies(
        self, mock_data_provider_uptrend: Mock, position_size_calculator: PositionSizeCalculator
    ) -> None:
        """Initializes with data provider and position calculator."""
        handler = TrendFollowingStrategyHandler(
            mock_data_provider_uptrend, position_size_calculator
        )

        assert handler.data_provider is mock_data_provider_uptrend
        assert handler.position_size_calculator is position_size_calculator


class TestSignalGeneration:
    """Test trend following signal generation."""

    def test_generates_buy_signal_for_ma_alignment(
        self,
        trend_following_handler: TrendFollowingStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Generates BUY signal when MAs are aligned (10>30>50)."""
        signals = trend_following_handler.generate_signals(
            ["AAPL"], tier_config, portfolio_snapshot
        )

        assert len(signals) > 0
        signal = signals[0]
        assert signal.symbol == "AAPL"
        assert signal.action == "BUY"
        assert signal.strategy_source == "trend_following"
        assert "uptrend" in signal.reasoning.lower()

    def test_no_signal_for_insufficient_data(
        self,
        trend_following_handler: TrendFollowingStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Does not generate signals when historical data is too short."""
        short_data = Mock()
        short_data.__len__ = Mock(return_value=30)
        short_data.data = {"Close": [100.0] * 30}
        trend_following_handler.data_provider.get_historical_data = Mock(return_value=short_data)

        signals = trend_following_handler.generate_signals(
            ["AAPL"], tier_config, portfolio_snapshot
        )

        assert signals == []

    def test_no_signal_for_misaligned_mas(
        self,
        trend_following_handler: TrendFollowingStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Does not generate signals when MAs are not aligned."""
        # Flat price data - no trend
        flat_data = Mock()
        flat_data.__len__ = Mock(return_value=100)
        flat_data.data = {"Close": [100.0] * 100}
        trend_following_handler.data_provider.get_historical_data = Mock(return_value=flat_data)

        signals = trend_following_handler.generate_signals(
            ["AAPL"], tier_config, portfolio_snapshot
        )

        assert signals == []

    def test_confidence_based_on_trend_strength(
        self,
        trend_following_handler: TrendFollowingStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Signal confidence scales with trend strength."""
        signals = trend_following_handler.generate_signals(
            ["AAPL"], tier_config, portfolio_snapshot
        )

        assert len(signals) > 0
        signal = signals[0]
        # Strong trend should have meaningful confidence
        assert 0.3 < signal.confidence <= 0.9

    def test_position_size_calculated(
        self,
        trend_following_handler: TrendFollowingStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Position size is calculated for generated signals."""
        signals = trend_following_handler.generate_signals(
            ["AAPL"], tier_config, portfolio_snapshot
        )

        assert len(signals) > 0
        signal = signals[0]
        assert signal.target_position_size > 0
        assert signal.target_position_size >= tier_config.min_position_size

    def test_processes_multiple_symbols(
        self,
        trend_following_handler: TrendFollowingStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Processes multiple symbols and generates signals."""
        signals = trend_following_handler.generate_signals(
            ["AAPL", "GOOGL", "MSFT"], tier_config, portfolio_snapshot
        )

        # Should generate signals for trending symbols
        assert len(signals) >= 1


class TestErrorHandling:
    """Test error handling."""

    def test_handles_data_provider_exceptions(
        self,
        trend_following_handler: TrendFollowingStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Handles data provider exceptions gracefully."""
        trend_following_handler.data_provider.get_historical_data = Mock(
            side_effect=Exception("API error")
        )

        signals = trend_following_handler.generate_signals(
            ["AAPL"], tier_config, portfolio_snapshot
        )

        assert signals == []

    def test_handles_missing_price_data(
        self,
        trend_following_handler: TrendFollowingStrategyHandler,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Handles missing or invalid price data."""
        bad_data = Mock()
        bad_data.__len__ = Mock(return_value=100)
        bad_data.data = {"Close": [None] * 100}
        trend_following_handler.data_provider.get_historical_data = Mock(return_value=bad_data)

        signals = trend_following_handler.generate_signals(
            ["AAPL"], tier_config, portfolio_snapshot
        )

        assert signals == []

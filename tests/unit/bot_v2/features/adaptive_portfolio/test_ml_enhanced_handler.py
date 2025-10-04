"""Tests for MLEnhancedStrategyHandler.

This module tests the ML-enhanced strategy handler's signal generation
which wraps and enhances momentum signals.

Critical behaviors tested:
- Enhancement of high-confidence momentum signals
- Confidence boosting (1.2x multiplier)
- Filtering of low-confidence signals
- Proper delegation to momentum handler
- Confidence caps at 0.95
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from bot_v2.features.adaptive_portfolio.strategy_handlers.ml_enhanced import (
    MLEnhancedStrategyHandler,
)
from bot_v2.features.adaptive_portfolio.types import (
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
def tier_config() -> TierConfig:
    """Create sample tier configuration."""
    return TierConfig(
        name="micro",
        range=(0, 5000),
        positions=PositionConstraints(1, 3, 2, 100.0),
        min_position_size=100.0,
        strategies=["ml_enhanced"],
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
def mock_momentum_handler() -> Mock:
    """Create mock momentum handler."""
    handler = Mock()
    handler.generate_signals = Mock(return_value=[])
    return handler


@pytest.fixture
def ml_enhanced_handler(mock_momentum_handler: Mock) -> MLEnhancedStrategyHandler:
    """Create ML-enhanced strategy handler."""
    return MLEnhancedStrategyHandler(mock_momentum_handler)


class TestMLEnhancedHandlerInitialization:
    """Test ML-enhanced handler initialization."""

    def test_initializes_with_momentum_handler(self, mock_momentum_handler: Mock) -> None:
        """Initializes with momentum handler dependency."""
        handler = MLEnhancedStrategyHandler(mock_momentum_handler)

        assert handler.momentum_handler is mock_momentum_handler


class TestSignalGeneration:
    """Test ML-enhanced signal generation."""

    def test_enhances_high_confidence_signals(
        self,
        ml_enhanced_handler: MLEnhancedStrategyHandler,
        mock_momentum_handler: Mock,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Enhances signals with confidence > 0.6."""
        # Mock momentum handler returns high-confidence signal
        high_conf_signal = TradingSignal(
            symbol="AAPL",
            action="BUY",
            confidence=0.7,
            target_position_size=500.0,
            stop_loss_pct=5.0,
            strategy_source="momentum",
            reasoning="Strong momentum",
        )
        mock_momentum_handler.generate_signals = Mock(return_value=[high_conf_signal])

        signals = ml_enhanced_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert len(signals) == 1
        enhanced = signals[0]
        assert enhanced.confidence == 0.7 * 1.2  # 0.84
        assert enhanced.strategy_source == "ml_enhanced"
        assert "ML-enhanced" in enhanced.reasoning

    def test_filters_low_confidence_signals(
        self,
        ml_enhanced_handler: MLEnhancedStrategyHandler,
        mock_momentum_handler: Mock,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Does not enhance signals with confidence <= 0.6."""
        # Mock momentum handler returns low-confidence signal
        low_conf_signal = TradingSignal(
            symbol="AAPL",
            action="BUY",
            confidence=0.5,
            target_position_size=500.0,
            stop_loss_pct=5.0,
            strategy_source="momentum",
            reasoning="Weak momentum",
        )
        mock_momentum_handler.generate_signals = Mock(return_value=[low_conf_signal])

        signals = ml_enhanced_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert signals == []

    def test_caps_confidence_at_95_percent(
        self,
        ml_enhanced_handler: MLEnhancedStrategyHandler,
        mock_momentum_handler: Mock,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Caps enhanced confidence at 0.95."""
        # Very high confidence signal (0.9 * 1.2 = 1.08, should cap at 0.95)
        very_high_conf = TradingSignal(
            symbol="AAPL",
            action="BUY",
            confidence=0.9,
            target_position_size=500.0,
            stop_loss_pct=5.0,
            strategy_source="momentum",
            reasoning="Very strong momentum",
        )
        mock_momentum_handler.generate_signals = Mock(return_value=[very_high_conf])

        signals = ml_enhanced_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert len(signals) == 1
        assert signals[0].confidence == 0.95

    def test_preserves_signal_attributes(
        self,
        ml_enhanced_handler: MLEnhancedStrategyHandler,
        mock_momentum_handler: Mock,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Preserves original signal attributes except confidence and source."""
        original_signal = TradingSignal(
            symbol="GOOGL",
            action="BUY",
            confidence=0.7,
            target_position_size=750.0,
            stop_loss_pct=4.5,
            strategy_source="momentum",
            reasoning="Original reasoning",
        )
        mock_momentum_handler.generate_signals = Mock(return_value=[original_signal])

        signals = ml_enhanced_handler.generate_signals(["GOOGL"], tier_config, portfolio_snapshot)

        enhanced = signals[0]
        assert enhanced.symbol == "GOOGL"
        assert enhanced.action == "BUY"
        assert enhanced.target_position_size == 750.0
        assert enhanced.stop_loss_pct == 4.5

    def test_processes_multiple_signals(
        self,
        ml_enhanced_handler: MLEnhancedStrategyHandler,
        mock_momentum_handler: Mock,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Processes multiple momentum signals."""
        signals_in = [
            TradingSignal("AAPL", "BUY", 0.7, 500.0, 5.0, "momentum", "Signal 1"),
            TradingSignal("GOOGL", "BUY", 0.65, 600.0, 5.0, "momentum", "Signal 2"),
            TradingSignal("MSFT", "BUY", 0.5, 400.0, 5.0, "momentum", "Signal 3"),  # Too low
        ]
        mock_momentum_handler.generate_signals = Mock(return_value=signals_in)

        signals = ml_enhanced_handler.generate_signals(
            ["AAPL", "GOOGL", "MSFT"], tier_config, portfolio_snapshot
        )

        # Should enhance first two, filter out third
        assert len(signals) == 2
        assert {s.symbol for s in signals} == {"AAPL", "GOOGL"}

    def test_delegates_to_momentum_handler(
        self,
        ml_enhanced_handler: MLEnhancedStrategyHandler,
        mock_momentum_handler: Mock,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Delegates signal generation to momentum handler."""
        ml_enhanced_handler.generate_signals(["AAPL", "GOOGL"], tier_config, portfolio_snapshot)

        mock_momentum_handler.generate_signals.assert_called_once_with(
            ["AAPL", "GOOGL"], tier_config, portfolio_snapshot
        )

    def test_returns_empty_when_no_momentum_signals(
        self,
        ml_enhanced_handler: MLEnhancedStrategyHandler,
        mock_momentum_handler: Mock,
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> None:
        """Returns empty list when momentum handler produces no signals."""
        mock_momentum_handler.generate_signals = Mock(return_value=[])

        signals = ml_enhanced_handler.generate_signals(["AAPL"], tier_config, portfolio_snapshot)

        assert signals == []

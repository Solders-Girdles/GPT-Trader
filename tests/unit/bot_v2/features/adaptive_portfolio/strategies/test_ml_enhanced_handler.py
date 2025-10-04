"""Tests for MLEnhancedStrategyHandler."""

from __future__ import annotations

from unittest.mock import Mock

from bot_v2.features.adaptive_portfolio.strategy_handlers.ml_enhanced import (
    MLEnhancedStrategyHandler,
)
from bot_v2.features.adaptive_portfolio.types import TradingSignal


def test_enhances_high_confidence_signals(
    micro_tier_config,
    portfolio_snapshot,
) -> None:
    momentum_handler = Mock(spec=["generate_signals"])
    momentum_handler.generate_signals.return_value = [
        TradingSignal("AAA", "BUY", 0.7, 150.0, 5.0, "momentum", "base"),
        TradingSignal("BBB", "BUY", 0.8, 180.0, 5.0, "momentum", "base"),
    ]

    handler = MLEnhancedStrategyHandler(momentum_handler)

    signals = handler.generate_signals(["AAA", "BBB"], micro_tier_config, portfolio_snapshot)

    assert len(signals) == 2
    confidences = [signal.confidence for signal in signals]
    assert confidences == [0.84, 0.95]
    for signal in signals:
        assert signal.strategy_source == "ml_enhanced"
        assert signal.target_position_size in (150.0, 180.0)


def test_filters_low_confidence_signals(
    micro_tier_config,
    portfolio_snapshot,
) -> None:
    momentum_handler = Mock(spec=["generate_signals"])
    momentum_handler.generate_signals.return_value = [
        TradingSignal("AAA", "BUY", 0.55, 150.0, 5.0, "momentum", "base"),
        TradingSignal("BBB", "BUY", 0.45, 150.0, 5.0, "momentum", "base"),
    ]

    handler = MLEnhancedStrategyHandler(momentum_handler)

    signals = handler.generate_signals(["AAA", "BBB"], micro_tier_config, portfolio_snapshot)

    assert len(signals) == 0


def test_returns_empty_when_momentum_returns_none(
    micro_tier_config,
    portfolio_snapshot,
) -> None:
    momentum_handler = Mock(spec=["generate_signals"])
    momentum_handler.generate_signals.return_value = []

    handler = MLEnhancedStrategyHandler(momentum_handler)

    signals = handler.generate_signals(["AAA"], micro_tier_config, portfolio_snapshot)

    assert signals == []

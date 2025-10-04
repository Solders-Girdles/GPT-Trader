"""Tests for MomentumStrategyHandler."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from bot_v2.features.adaptive_portfolio.strategy_handlers.momentum import MomentumStrategyHandler
from bot_v2.features.adaptive_portfolio.types import TradingSignal


def test_generates_buy_signal_for_strong_momentum(
    simple_frame_factory,
    micro_tier_config,
    portfolio_snapshot,
    position_size_calculator_mock,
) -> None:
    data_provider = Mock()
    prices = [100.0 + i * 1.5 for i in range(60)]  # Strong uptrend
    data_provider.get_historical_data.return_value = simple_frame_factory(prices)

    handler = MomentumStrategyHandler(data_provider, position_size_calculator_mock)

    signals = handler.generate_signals(["AAPL"], micro_tier_config, portfolio_snapshot)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.strategy_source == "momentum"
    assert signal.action == "BUY"
    assert pytest.approx(signal.target_position_size) == round(signal.confidence * 1000, 2)
    position_size_calculator_mock.calculate.assert_called_once()


def test_returns_empty_when_history_too_short(
    simple_frame_factory,
    micro_tier_config,
    portfolio_snapshot,
    position_size_calculator_mock,
) -> None:
    data_provider = Mock()
    data_provider.get_historical_data.return_value = simple_frame_factory([100.0] * 10)

    handler = MomentumStrategyHandler(data_provider, position_size_calculator_mock)

    signals = handler.generate_signals(["AAPL"], micro_tier_config, portfolio_snapshot)

    assert signals == []
    position_size_calculator_mock.calculate.assert_not_called()


def test_handles_data_provider_exception(
    micro_tier_config,
    portfolio_snapshot,
    position_size_calculator_mock,
) -> None:
    data_provider = Mock()
    data_provider.get_historical_data.side_effect = Exception("API error")

    handler = MomentumStrategyHandler(data_provider, position_size_calculator_mock)

    signals = handler.generate_signals(["AAPL"], micro_tier_config, portfolio_snapshot)

    assert signals == []

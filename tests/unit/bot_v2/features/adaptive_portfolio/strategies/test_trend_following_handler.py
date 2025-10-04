"""Tests for TrendFollowingStrategyHandler."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from bot_v2.features.adaptive_portfolio.strategy_handlers.trend_following import (
    TrendFollowingStrategyHandler,
)


def test_generates_buy_signal_for_aligned_trend(
    simple_frame_factory,
    micro_tier_config,
    portfolio_snapshot,
    position_size_calculator_mock,
) -> None:
    data_provider = Mock()
    prices = [100.0 + i for i in range(120)]  # Strong, steady uptrend
    data_provider.get_historical_data.return_value = simple_frame_factory(prices)

    handler = TrendFollowingStrategyHandler(data_provider, position_size_calculator_mock)

    signals = handler.generate_signals(["TSLA"], micro_tier_config, portfolio_snapshot)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.strategy_source == "trend_following"
    assert signal.action == "BUY"
    assert pytest.approx(signal.target_position_size) == round(signal.confidence * 1000, 2)
    position_size_calculator_mock.calculate.assert_called_once()


def test_returns_empty_when_history_insufficient(
    simple_frame_factory,
    micro_tier_config,
    portfolio_snapshot,
    position_size_calculator_mock,
) -> None:
    data_provider = Mock()
    data_provider.get_historical_data.return_value = simple_frame_factory([100.0] * 40)

    handler = TrendFollowingStrategyHandler(data_provider, position_size_calculator_mock)

    signals = handler.generate_signals(["TSLA"], micro_tier_config, portfolio_snapshot)

    assert signals == []
    position_size_calculator_mock.calculate.assert_not_called()


def test_handles_data_provider_exception(
    micro_tier_config,
    portfolio_snapshot,
    position_size_calculator_mock,
) -> None:
    data_provider = Mock()
    data_provider.get_historical_data.side_effect = Exception("API error")

    handler = TrendFollowingStrategyHandler(data_provider, position_size_calculator_mock)

    signals = handler.generate_signals(["TSLA"], micro_tier_config, portfolio_snapshot)

    assert signals == []

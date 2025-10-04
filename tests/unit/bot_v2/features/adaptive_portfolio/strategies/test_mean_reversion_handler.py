"""Tests for MeanReversionStrategyHandler."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from bot_v2.features.adaptive_portfolio.strategy_handlers.mean_reversion import (
    MeanReversionStrategyHandler,
)


def test_generates_buy_signal_for_oversold_asset(
    simple_frame_factory,
    micro_tier_config,
    portfolio_snapshot,
    position_size_calculator_mock,
) -> None:
    data_provider = Mock()
    prices = [100.0] * 59 + [70.0]  # Last close far below 20-day average
    data_provider.get_historical_data.return_value = simple_frame_factory(prices)

    handler = MeanReversionStrategyHandler(data_provider, position_size_calculator_mock)

    signals = handler.generate_signals(["ETH"], micro_tier_config, portfolio_snapshot)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.strategy_source == "mean_reversion"
    assert signal.action == "BUY"
    assert pytest.approx(signal.target_position_size) == round(signal.confidence * 1000, 2)
    position_size_calculator_mock.calculate.assert_called_once()


def test_returns_empty_when_standard_deviation_zero(
    simple_frame_factory,
    micro_tier_config,
    portfolio_snapshot,
    position_size_calculator_mock,
) -> None:
    data_provider = Mock()
    data_provider.get_historical_data.return_value = simple_frame_factory([100.0] * 60)

    handler = MeanReversionStrategyHandler(data_provider, position_size_calculator_mock)

    signals = handler.generate_signals(["ETH"], micro_tier_config, portfolio_snapshot)

    assert signals == []
    position_size_calculator_mock.calculate.assert_not_called()


def test_handles_data_provider_exception(
    micro_tier_config,
    portfolio_snapshot,
    position_size_calculator_mock,
) -> None:
    data_provider = Mock()
    data_provider.get_historical_data.side_effect = Exception("API error")

    handler = MeanReversionStrategyHandler(data_provider, position_size_calculator_mock)

    signals = handler.generate_signals(["ETH"], micro_tier_config, portfolio_snapshot)

    assert signals == []

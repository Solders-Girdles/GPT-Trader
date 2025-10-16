from __future__ import annotations

import pandas as pd
import pytest

from bot_v2.features.optimize.strategies import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    SimpleMAStrategy,
    VolatilityStrategy,
    create_local_strategy,
    get_strategy_params,
)


def _frame(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(values), freq="1h")
    return pd.DataFrame(
        {
            "open": values,
            "high": [v + 1 for v in values],
            "low": [v - 1 for v in values],
            "close": values,
            "volume": [10_000 + i for i in range(len(values))],
        },
        index=index,
    )


def test_simple_ma_strategy_generates_cross_signals():
    prices = [10, 10, 10, 10, 10, 11, 12, 13, 14, 13, 12, 11, 10]
    data = _frame(prices)
    strat = SimpleMAStrategy(fast_period=3, slow_period=5)
    signals = strat.generate_signals(data)

    assert 1 in signals.values
    assert -1 in signals.values
    assert strat.get_required_periods() == 6


def test_momentum_strategy_holds_and_exits():
    prices = [10, 10.02, 10.03, 10.04, 10.05, 10.3, 10.32, 10.34, 10.36, 10.38]
    data = _frame(prices)
    strat = MomentumStrategy(lookback=2, threshold=0.02, hold_period=2)
    signals = strat.generate_signals(data)

    entries = signals[signals == 1]
    exits = signals[signals == -1]
    assert len(entries) == 1
    assert len(exits) == 1
    entry_idx = entries.index[0]
    exit_idx = exits.index[0]
    assert (exit_idx - entry_idx).components.days == 0
    assert strat.get_required_periods() == 3


def test_mean_reversion_strategy_enters_and_exits():
    prices = [10] * 40
    prices[20] = 6  # oversold
    prices[30] = 14  # overbought
    data = _frame(prices)
    strat = MeanReversionStrategy(period=10, entry_std=2.0, exit_std=0.5)
    signals = strat.generate_signals(data)

    assert (signals == 1).any() or (signals == -1).any()
    assert strat.get_required_periods() == 10


def test_volatility_strategy_filters_high_volatility():
    prices = [100 + ((-1) ** i) * 0.3 for i in range(120)]
    data = _frame(prices)
    strat = VolatilityStrategy(vol_period=5, vol_threshold=1e-6, trend_period=10)
    signals = strat.generate_signals(data)

    assert signals.abs().sum() == 0
    assert strat.get_required_periods() == 11

    smooth_prices = [100 + i * 0.005 for i in range(120)]
    smooth_data = _frame(smooth_prices)
    signals_low_vol = strat.generate_signals(smooth_data)
    assert signals_low_vol.abs().sum() > 0


def test_breakout_strategy_confirms_and_stops():
    prices = [10] * 30
    prices += [12, 13, 14, 15, 15.5, 15.8, 15.9, 16, 15.1, 14.2, 13.5]
    data = _frame(prices)
    strat = BreakoutStrategy(lookback=5, confirm_bars=1, stop_loss=0.05)
    signals = strat.generate_signals(data)

    assert (signals == 1).any()
    assert strat.get_required_periods() == 6


def test_strategy_factory_and_params():
    strategy = create_local_strategy("Momentum", lookback=5)
    assert isinstance(strategy, MomentumStrategy)
    params = get_strategy_params("Momentum")
    assert "lookback" in params

    with pytest.raises(ValueError):
        create_local_strategy("Nonexistent")

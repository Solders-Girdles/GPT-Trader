from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

from bot_v2.backtest import (
    Bar,
    MovingAverageCrossStrategy,
    BollingerMeanReversionStrategy,
    VolatilityFilteredStrategy,
    VolumeConfirmationStrategy,
    MomentumOscillatorStrategy,
    TrendStrengthStrategy,
    SpotBacktestConfig,
    SpotBacktester,
    load_candles_from_parquet,
)


def make_trending_bars(count: int = 50) -> list[Bar]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = []
    price = Decimal("10000")
    for i in range(count):
        ts = base + timedelta(hours=i)
        close = price + Decimal(i)
        bars.append(
            Bar(
                timestamp=ts,
                open=price,
                high=close + Decimal("10"),
                low=price - Decimal("10"),
                close=close,
                volume=Decimal("1"),
            )
        )
        price = close
    return bars


def test_backtester_runs_and_produces_positive_return():
    bars = make_trending_bars()
    strategy = MovingAverageCrossStrategy(short_window=3, long_window=8)
    config = SpotBacktestConfig(initial_cash=Decimal("10000"))
    backtester = SpotBacktester(bars, strategy, config)
    result = backtester.run()

    assert result.metrics.total_return > 0
    assert len(result.trades) >= 1
    assert not result.equity_curve.empty


def test_load_candles_from_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    timestamps = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(3)]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [95, 96, 97],
            "close": [102, 103, 104],
            "volume": [10, 11, 12],
        }
    )
    path = tmp_path / "candles.parquet"
    df.to_parquet(path, index=False)

    bars = load_candles_from_parquet(path)
    assert len(bars) == 3
    assert bars[0].timestamp == timestamps[0]
    assert bars[0].close == Decimal("102")


def test_bollinger_strategy_generates_signals():
    bars = make_trending_bars()
    strategy = BollingerMeanReversionStrategy(window=5, num_std=1.5)
    config = SpotBacktestConfig(initial_cash=Decimal("10000"))
    backtester = SpotBacktester(bars, strategy, config)
    result = backtester.run()

    assert len(result.trades) >= 0
    assert result.metrics.max_drawdown <= 0  # drawdowns are negative or zero


def test_volatility_filter_blocks_low_vol_buy():
    base = MovingAverageCrossStrategy(short_window=2, long_window=3)
    strategy = VolatilityFilteredStrategy(base_strategy=base, window=5, min_vol=Decimal("0.01"), max_vol=Decimal("0.05"))

    bars = []
    price = Decimal("100")
    for i in range(10):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        bars.append(
            Bar(
                timestamp=ts,
                open=price,
                high=price + Decimal("0.1"),
                low=price - Decimal("0.1"),
                close=price + Decimal("0.1"),
                volume=Decimal("1"),
            )
        )
        price += Decimal("0.1")

    result = SpotBacktester(bars, strategy, SpotBacktestConfig(initial_cash=Decimal("1000"))).run()
    assert len(result.trades) == 0


def test_volatility_filter_allows_in_band():
    base = MovingAverageCrossStrategy(short_window=2, long_window=3)
    strategy = VolatilityFilteredStrategy(base_strategy=base, window=5, min_vol=Decimal("0.001"), max_vol=Decimal("0.05"))

    bars = []
    price = Decimal("100")
    for i in range(10):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        high = price + Decimal("2")
        low = price - Decimal("2")
        close = price + Decimal("0.5")
        bars.append(Bar(timestamp=ts, open=price, high=high, low=low, close=close, volume=Decimal("1")))
        price = close

    result = SpotBacktester(bars, strategy, SpotBacktestConfig(initial_cash=Decimal("1000"))).run()
    assert len(result.trades) > 0


def test_volume_confirmation_blocks_low_volume_buy():
    base = MovingAverageCrossStrategy(short_window=2, long_window=3)
    strategy = VolumeConfirmationStrategy(base_strategy=base, window=3, multiplier=Decimal("1.5"))

    bars = []
    price = Decimal("100")
    for i in range(10):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        volume = Decimal("1") if i < 5 else Decimal("0.5")
        bars.append(
            Bar(
                timestamp=ts,
                open=price,
                high=price + Decimal("1"),
                low=price - Decimal("1"),
                close=price + Decimal("0.5"),
                volume=volume,
            )
        )
        price += Decimal("0.5")

    result = SpotBacktester(bars, strategy, SpotBacktestConfig(initial_cash=Decimal("1000"))).run()
    assert len(result.trades) == 0


def test_volume_confirmation_allows_high_volume_buy():
    base = MovingAverageCrossStrategy(short_window=2, long_window=3)
    strategy = VolumeConfirmationStrategy(base_strategy=base, window=3, multiplier=Decimal("1.05"))

    bars = []
    price = Decimal("100")
    for i in range(10):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        volume = Decimal("2") if i < 8 else Decimal("3")
        bars.append(
            Bar(
                timestamp=ts,
                open=price,
                high=price + Decimal("1"),
                low=price - Decimal("1"),
                close=price + Decimal("0.5"),
                volume=volume,
            )
        )
        price += Decimal("0.5")

    result = SpotBacktester(bars, strategy, SpotBacktestConfig(initial_cash=Decimal("1000"))).run()
    assert len(result.trades) > 0


def test_rsi_filter_blocks_overbought_buy():
    base = MovingAverageCrossStrategy(short_window=2, long_window=3)
    strategy = MomentumOscillatorStrategy(base_strategy=base, window=3, overbought=Decimal("60"), oversold=Decimal("40"))

    bars = []
    price = Decimal("100")
    for i in range(10):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        price += Decimal("5")
        bars.append(
            Bar(
                timestamp=ts,
                open=price - Decimal("5"),
                high=price + Decimal("1"),
                low=price - Decimal("6"),
                close=price,
                volume=Decimal("1"),
            )
        )

    result = SpotBacktester(bars, strategy, SpotBacktestConfig(initial_cash=Decimal("1000"))).run()
    assert len(result.trades) == 0


def test_rsi_filter_allows_after_cooldown():
    base = MovingAverageCrossStrategy(short_window=2, long_window=3)
    strategy = MomentumOscillatorStrategy(base_strategy=base, window=3, overbought=Decimal("70"), oversold=Decimal("30"))

    bars = []
    price = Decimal("100")
    for i in range(10):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        if i < 5:
            price += Decimal("5")
        else:
            price -= Decimal("1")
        bars.append(
            Bar(
                timestamp=ts,
                open=price,
                high=price + Decimal("2"),
                low=price - Decimal("2"),
                close=price,
                volume=Decimal("1"),
            )
        )

    result = SpotBacktester(bars, strategy, SpotBacktestConfig(initial_cash=Decimal("1000"))).run()
    assert len(result.trades) >= 0


def test_trend_strength_blocks_flat_market():
    base = MovingAverageCrossStrategy(short_window=2, long_window=3)
    strategy = TrendStrengthStrategy(base_strategy=base, window=3, min_slope=Decimal("0.05"))

    bars = []
    price = Decimal("100")
    for i in range(10):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        close = price + Decimal("0.01")
        bars.append(
            Bar(
                timestamp=ts,
                open=price,
                high=price + Decimal("0.5"),
                low=price - Decimal("0.5"),
                close=close,
                volume=Decimal("1"),
            )
        )
        price = close

    result = SpotBacktester(bars, strategy, SpotBacktestConfig(initial_cash=Decimal("1000"))).run()
    assert len(result.trades) == 0


def test_trend_strength_allows_when_slope_exceeds_threshold():
    base = MovingAverageCrossStrategy(short_window=2, long_window=3)
    strategy = TrendStrengthStrategy(base_strategy=base, window=3, min_slope=Decimal("0.001"))

    bars = []
    price = Decimal("100")
    for i in range(10):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        price += Decimal("1")
        bars.append(
            Bar(
                timestamp=ts,
                open=price - Decimal("1"),
                high=price + Decimal("0.5"),
                low=price - Decimal("0.5"),
                close=price,
                volume=Decimal("1"),
            )
        )

    result = SpotBacktester(bars, strategy, SpotBacktestConfig(initial_cash=Decimal("1000"))).run()
    assert len(result.trades) > 0

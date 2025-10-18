from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from gpt_trader.domain import Bar, Signal, Strategy
from gpt_trader.strategy.simple import (
    DonchianBreakoutStrategy,
    MovingAverageCrossStrategy,
    VolatilityScaledStrategy,
)


def _bar(price: float, ts: datetime) -> Bar:
    value = Decimal(str(price))
    return Bar(
        symbol="BTC-USD",
        timestamp=ts,
        open=value,
        high=value,
        low=value,
        close=value,
        volume=Decimal("1"),
    )


def test_donchian_breakout_buy_signal() -> None:
    """Breakout above channel should produce BUY with metadata."""
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = [
        _bar(price, ts + timedelta(hours=idx))
        for idx, price in enumerate([100, 101, 99, 102, 98, 103, 101, 104])
    ]
    strategy = DonchianBreakoutStrategy(lookback=5)

    signal = strategy.decide(bars)

    assert signal.action == "BUY"
    assert signal.confidence > 0
    assert signal.metadata is not None
    assert signal.metadata["reason"] == "breakout_above"
    assert signal.metadata["channel_high"] == pytest.approx(103)
    assert signal.metadata["channel_low"] == pytest.approx(98)


def test_donchian_breakout_sell_signal() -> None:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100, 95, 97, 96, 94, 93, 92, 90]
    bars = [_bar(price, ts + timedelta(hours=i)) for i, price in enumerate(prices)]
    strategy = DonchianBreakoutStrategy(lookback=6)

    signal = strategy.decide(bars)

    assert signal.action == "SELL"
    assert signal.metadata["reason"] == "breakout_below"
    assert signal.confidence > 0


def test_donchian_breakout_needs_history() -> None:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = [_bar(100, ts)]
    strategy = DonchianBreakoutStrategy(lookback=3)

    signal = strategy.decide(bars)

    assert signal.action == "HOLD"
    assert signal.confidence == 0.0
    assert signal.metadata["reason"] == "insufficient_history"


class _ConstantStrategy(Strategy):
    def __init__(self, action: str, confidence: float = 1.0) -> None:
        self._action = action
        self._confidence = confidence

    def decide(self, bars: list[Bar]) -> Signal:
        return Signal(symbol=bars[-1].symbol, action=self._action, confidence=self._confidence)


def test_volatility_scaled_strategy_scales_confidence() -> None:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100, 101, 102, 103, 104, 105]
    bars = [_bar(price, ts + timedelta(hours=i)) for i, price in enumerate(prices)]
    base = _ConstantStrategy("BUY", confidence=1.0)
    strategy = VolatilityScaledStrategy(base, vol_window=3, target_vol=0.01)

    signal = strategy.decide(bars)

    assert signal.action == "BUY"
    assert 0 < signal.confidence <= 1.0
    assert signal.metadata is not None
    scaled_meta = signal.metadata["volatility_scaled"]
    assert scaled_meta["realized_vol"] >= 0
    assert scaled_meta["scaling"] <= 1.0


def test_volatility_scaled_strategy_handles_low_history() -> None:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = [_bar(100, ts)]
    strategy = VolatilityScaledStrategy(_ConstantStrategy("BUY"))

    signal = strategy.decide(bars)

    assert signal.action == "BUY"
    assert signal.confidence == 0.0
    assert signal.metadata["reason"] == "insufficient_history"

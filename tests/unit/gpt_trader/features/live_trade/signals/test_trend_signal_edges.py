from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.live_trade.signals.protocol import StrategyContext
from gpt_trader.features.live_trade.signals.trend import TrendSignal, TrendSignalConfig


def _make_context(prices: list[Decimal]) -> StrategyContext:
    return StrategyContext(
        symbol="BTC-USD",
        current_mark=prices[-1] if prices else Decimal("0"),
        position_state=None,
        recent_marks=prices,
        equity=Decimal("1000"),
        product=None,
        candles=None,
        market_data=None,
    )


def test_insufficient_data_returns_neutral() -> None:
    signal = TrendSignal(TrendSignalConfig(fast_period=2, slow_period=3))

    output = signal.generate(_make_context([Decimal("1"), Decimal("2")]))

    assert output.strength == 0.0
    assert output.confidence == 0.0
    assert output.metadata["reason"] == "insufficient_data"


def test_calculation_error_when_fast_exceeds_available() -> None:
    signal = TrendSignal(TrendSignalConfig(fast_period=5, slow_period=3))

    output = signal.generate(_make_context([Decimal("1"), Decimal("1"), Decimal("1")]))

    assert output.strength == 0.0
    assert output.confidence == 0.0
    assert output.metadata["reason"] == "calculation_error"


def test_bullish_crossover_signal() -> None:
    signal = TrendSignal(TrendSignalConfig(fast_period=2, slow_period=3))

    output = signal.generate(
        _make_context([Decimal("1"), Decimal("1"), Decimal("1"), Decimal("10")])
    )

    assert output.strength == 1.0
    assert output.confidence == 0.8
    assert output.metadata["reason"] == "bullish_crossover"


def test_bearish_crossover_signal() -> None:
    signal = TrendSignal(TrendSignalConfig(fast_period=2, slow_period=3))

    output = signal.generate(
        _make_context([Decimal("10"), Decimal("10"), Decimal("10"), Decimal("1")])
    )

    assert output.strength == -1.0
    assert output.confidence == 0.8
    assert output.metadata["reason"] == "bearish_crossover"

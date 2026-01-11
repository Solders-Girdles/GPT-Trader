from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.live_trade.signals.momentum import (
    MomentumSignal,
    MomentumSignalConfig,
)
from gpt_trader.features.live_trade.signals.protocol import StrategyContext


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
    signal = MomentumSignal(MomentumSignalConfig(period=3))

    output = signal.generate(_make_context([Decimal("10"), Decimal("9"), Decimal("8")]))

    assert output.strength == 0.0
    assert output.confidence == 0.0
    assert output.metadata["reason"] == "insufficient_data"


def test_oversold_strength_and_confidence() -> None:
    signal = MomentumSignal(MomentumSignalConfig(period=3))

    output = signal.generate(
        _make_context([Decimal("10"), Decimal("9"), Decimal("8"), Decimal("7")])
    )

    assert output.strength == 1.0
    assert output.metadata["reason"] == "oversold"
    assert output.confidence == 1.0


def test_overbought_strength_and_confidence() -> None:
    signal = MomentumSignal(MomentumSignalConfig(period=3))

    output = signal.generate(
        _make_context([Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4")])
    )

    assert output.strength == -1.0
    assert output.metadata["reason"] == "overbought"
    assert output.confidence == 1.0

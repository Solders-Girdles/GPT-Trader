from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.live_trade.signals.mean_reversion import (
    MeanReversionSignal,
    MeanReversionSignalConfig,
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
    signal = MeanReversionSignal(MeanReversionSignalConfig(window=5))

    output = signal.generate(
        _make_context([Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100")])
    )

    assert output.strength == 0.0
    assert output.confidence == 0.0
    assert output.metadata["reason"] == "insufficient_data"


def test_zero_volatility_returns_reason() -> None:
    signal = MeanReversionSignal(MeanReversionSignalConfig(window=5))

    output = signal.generate(
        _make_context(
            [
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
            ]
        )
    )

    assert output.strength == 0.0
    assert output.confidence == 0.0
    assert output.metadata["reason"] == "zero_volatility"


def test_oversold_z_score_signal() -> None:
    config = MeanReversionSignalConfig(window=5, z_entry_threshold=1.0)
    signal = MeanReversionSignal(config)

    output = signal.generate(
        _make_context(
            [
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("80"),
            ]
        )
    )

    assert output.strength == 1.0
    assert output.metadata["reason"] == "oversold_z_score"


def test_overbought_z_score_signal() -> None:
    config = MeanReversionSignalConfig(window=5, z_entry_threshold=1.0)
    signal = MeanReversionSignal(config)

    output = signal.generate(
        _make_context(
            [
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("100"),
                Decimal("120"),
            ]
        )
    )

    assert output.strength == -1.0
    assert output.metadata["reason"] == "overbought_z_score"


def test_mean_reverted_path() -> None:
    config = MeanReversionSignalConfig(
        window=5,
        z_entry_threshold=1.0,
        z_exit_threshold=5.0,
    )
    signal = MeanReversionSignal(config)

    output = signal.generate(
        _make_context(
            [
                Decimal("100"),
                Decimal("101"),
                Decimal("100"),
                Decimal("101"),
                Decimal("100"),
            ]
        )
    )

    assert output.metadata["reason"] == "mean_reverted"
    assert output.confidence == 0.5

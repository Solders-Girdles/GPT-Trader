from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.combiners.regime import (
    RegimeAwareCombiner,
    RegimeCombinerConfig,
)
from gpt_trader.features.live_trade.signals.protocol import StrategyContext
from gpt_trader.features.live_trade.signals.types import SignalOutput, SignalType


def _signal(name: str, signal_type: SignalType, strength: float, confidence: float) -> SignalOutput:
    return SignalOutput(
        name=name,
        type=signal_type,
        strength=strength,
        confidence=confidence,
        metadata={},
    )


def _context() -> StrategyContext:
    return StrategyContext(
        symbol="BTC-USD",
        current_mark=Decimal("100"),
        position_state=None,
        recent_marks=[],
        equity=Decimal("1000"),
        product=None,
        candles=[],
        market_data=None,
    )


def test_trending_regime_weights(monkeypatch) -> None:
    combiner = RegimeAwareCombiner(RegimeCombinerConfig())
    monkeypatch.setattr(combiner, "_calculate_adx", lambda context: Decimal("30"))

    signals = [
        _signal("trend", SignalType.TREND, 0.8, 0.6),
        _signal("mean", SignalType.MEAN_REVERSION, -0.4, 0.9),
    ]

    result = combiner.combine(signals, _context())

    assert result.metadata["regime"] == "trending"
    assert result.strength == pytest.approx(0.8)
    assert result.confidence == pytest.approx(0.6)


def test_ranging_regime_weights(monkeypatch) -> None:
    combiner = RegimeAwareCombiner(RegimeCombinerConfig())
    monkeypatch.setattr(combiner, "_calculate_adx", lambda context: Decimal("10"))

    signals = [
        _signal("trend", SignalType.TREND, 0.8, 0.6),
        _signal("mean", SignalType.MEAN_REVERSION, -0.4, 0.9),
    ]

    result = combiner.combine(signals, _context())

    assert result.metadata["regime"] == "ranging"
    assert result.strength == pytest.approx(-0.4)
    assert result.confidence == pytest.approx(0.9)


def test_neutral_blend_when_adx_none(monkeypatch) -> None:
    combiner = RegimeAwareCombiner(RegimeCombinerConfig())
    monkeypatch.setattr(combiner, "_calculate_adx", lambda context: None)

    signals = [
        _signal("trend", SignalType.TREND, 0.8, 0.6),
        _signal("mean", SignalType.MEAN_REVERSION, -0.4, 0.9),
    ]

    result = combiner.combine(signals, _context())

    assert combiner._current_regime == "neutral"
    assert result.metadata["regime"] == "neutral"
    assert result.strength == pytest.approx(0.2)
    assert result.confidence == pytest.approx(0.75)


def test_hysteresis_keeps_trending(monkeypatch) -> None:
    combiner = RegimeAwareCombiner(RegimeCombinerConfig())
    values = iter([Decimal("30"), Decimal("22")])
    monkeypatch.setattr(combiner, "_calculate_adx", lambda context: next(values))

    signals = [_signal("trend", SignalType.TREND, 0.6, 0.4)]

    first = combiner.combine(signals, _context())
    second = combiner.combine(signals, _context())

    assert first.metadata["regime"] == "trending"
    assert second.metadata["regime"] == "trending"
    assert combiner._current_regime == "trending"


def test_empty_signals_return_zeroes(monkeypatch) -> None:
    combiner = RegimeAwareCombiner(RegimeCombinerConfig())
    monkeypatch.setattr(combiner, "_calculate_adx", lambda context: Decimal("30"))

    result = combiner.combine([], _context())

    assert result.strength == 0.0
    assert result.confidence == 0.0
    assert result.metadata["components"] == {}

from __future__ import annotations

from decimal import Decimal

from gpt_trader.core import Action
from gpt_trader.features.live_trade.strategies.perps_baseline.stateful import (
    StatefulBaselineStrategy,
)
from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import (
    PerpsStrategyConfig,
)


class SimpleConfig:
    short_ma_period = 2
    long_ma_period = 3
    rsi_period = 2


def _make_strategy(**overrides) -> StatefulBaselineStrategy:
    config = PerpsStrategyConfig(
        short_ma_period=2,
        long_ma_period=3,
        rsi_period=2,
        min_confidence=0.1,
        **overrides,
    )
    return StatefulBaselineStrategy(config=config)


def test_parse_config_from_object() -> None:
    strategy = StatefulBaselineStrategy(config=SimpleConfig())

    assert strategy.config.short_ma_period == 2
    assert strategy.config.long_ma_period == 3
    assert strategy.config.rsi_period == 2


def test_kill_switch_returns_hold_with_indicator() -> None:
    strategy = _make_strategy(kill_switch_enabled=True)

    decision = strategy.decide(
        symbol="BTC-USD",
        current_mark=Decimal("100"),
        position_state=None,
        recent_marks=[],
        equity=Decimal("1000"),
        product=None,
    )

    assert decision.action == Action.HOLD
    assert decision.indicators["kill_switch"] is True


def test_warming_up_indicator() -> None:
    strategy = _make_strategy()

    decision = strategy.decide(
        symbol="BTC-USD",
        current_mark=Decimal("100"),
        position_state=None,
        recent_marks=[],
        equity=Decimal("1000"),
        product=None,
    )

    assert decision.action == Action.HOLD
    assert decision.indicators["warming_up"] is True
    assert decision.indicators["required"] == 3


def test_after_warmup_indicators_present() -> None:
    strategy = _make_strategy()
    symbol = "BTC-USD"

    for price in [Decimal("100"), Decimal("101"), Decimal("102")]:
        strategy.update(symbol, price)

    decision = strategy.decide(
        symbol=symbol,
        current_mark=Decimal("103"),
        position_state=None,
        recent_marks=[],
        equity=Decimal("1000"),
        product=None,
    )

    assert decision.indicators["stateful"] is True
    assert decision.indicators["rsi"] is not None
    assert decision.indicators["short_ma"] is not None
    assert decision.indicators["long_ma"] is not None


def test_serialize_deserialize_roundtrip() -> None:
    strategy = _make_strategy()
    symbol = "BTC-USD"

    for price in [Decimal("100"), Decimal("101"), Decimal("102")]:
        strategy.update(symbol, price)

    state = strategy.serialize_state()

    restored = _make_strategy()
    restored.deserialize_state(state)

    assert restored.serialize_state() == state

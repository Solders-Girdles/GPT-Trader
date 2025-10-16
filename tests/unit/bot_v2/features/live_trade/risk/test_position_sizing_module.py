from __future__ import annotations

from decimal import Decimal

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk.position_sizing import (
    ImpactAssessment,
    ImpactRequest,
    PositionSizer,
    PositionSizingAdvice,
    PositionSizingContext,
)


class StubEventStore:
    def __init__(self) -> None:
        self.metrics: list[tuple] = []

    def append_metric(self, *args, **kwargs) -> None:
        self.metrics.append((args, kwargs))


def test_position_sizer_reduce_only_returns_zero():
    config = RiskConfig(enable_dynamic_position_sizing=True)
    event_store = StubEventStore()
    sizer = PositionSizer(
        config=config,
        event_store=event_store,
        position_size_estimator=None,
        impact_estimator=None,
        is_reduce_only_mode=lambda: True,
    )

    context = PositionSizingContext(
        symbol="BTC-USD",
        side="buy",
        equity=Decimal("10000"),
        current_price=Decimal("20000"),
        strategy_name="test",
        method="intelligent",
        target_leverage=Decimal("2"),
    )

    advice = sizer.size_position(context)

    assert advice.reduce_only is True
    assert advice.target_quantity == Decimal("0")
    assert event_store.metrics  # metric emitted


def test_position_sizer_dynamic_estimator_used():
    config = RiskConfig(enable_dynamic_position_sizing=True)
    event_store = StubEventStore()

    expected_advice = PositionSizingAdvice(
        symbol="BTC-USD",
        side="sell",
        target_notional=Decimal("5000"),
        target_quantity=Decimal("0.25"),
        used_dynamic=True,
    )

    def estimator(context: PositionSizingContext) -> PositionSizingAdvice:
        assert context.symbol == "BTC-USD"
        return expected_advice

    sizer = PositionSizer(
        config=config,
        event_store=event_store,
        position_size_estimator=estimator,
        impact_estimator=None,
        is_reduce_only_mode=lambda: False,
    )

    context = PositionSizingContext(
        symbol="BTC-USD",
        side="sell",
        equity=Decimal("15000"),
        current_price=Decimal("20000"),
        strategy_name="momentum",
        method="dynamic",
        target_leverage=Decimal("1.5"),
    )

    advice = sizer.size_position(context)
    assert advice is expected_advice
    assert event_store.metrics


def test_position_sizer_fallback_when_estimator_fails():
    config = RiskConfig(enable_dynamic_position_sizing=True, position_sizing_multiplier=1.5)
    event_store = StubEventStore()

    def failing_estimator(context: PositionSizingContext) -> PositionSizingAdvice:
        raise RuntimeError("boom")

    sizer = PositionSizer(
        config=config,
        event_store=event_store,
        position_size_estimator=failing_estimator,
        impact_estimator=None,
        is_reduce_only_mode=lambda: False,
    )

    context = PositionSizingContext(
        symbol="BTC-USD",
        side="buy",
        equity=Decimal("8000"),
        current_price=Decimal("0"),
        strategy_name="fallback",
        method="notional",
        target_leverage=Decimal("2"),
        strategy_multiplier=1.5,
    )

    advice = sizer.size_position(context)

    assert advice.fallback_used is True
    assert advice.target_notional == Decimal("8000") * Decimal("2") * Decimal("1.5")
    assert advice.target_quantity == Decimal("0")
    assert event_store.metrics

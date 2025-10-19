"""Position sizing integration tests for LiveRiskManager."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from tests.support.event_store import RecordingEventStore

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk import (
    LiveRiskManager,
    PositionSizingAdvice,
    PositionSizingContext,
)


def _make_context(**overrides) -> PositionSizingContext:
    base = PositionSizingContext(
        symbol="BTC-PERP",
        side="buy",
        equity=Decimal("10000"),
        current_price=Decimal("50000"),
        strategy_name="TestStrategy",
        method="intelligent",
        target_leverage=Decimal("2"),
        product=None,
    )
    return base if not overrides else replace(base, **overrides)


def test_size_position_uses_dynamic_estimator():
    config = RiskConfig(enable_dynamic_position_sizing=True)
    store = RecordingEventStore()

    captured = {}

    def estimator(context: PositionSizingContext) -> PositionSizingAdvice:
        captured["context"] = context
        return PositionSizingAdvice(
            symbol=context.symbol,
            side=context.side,
            target_notional=Decimal("5000"),
            target_quantity=Decimal("0.1"),
            used_dynamic=True,
            reduce_only=False,
            reason="dynamic",
        )

    risk_manager = LiveRiskManager(
        config=config,
        event_store=store,
        position_size_estimator=estimator,
    )

    context = _make_context()
    advice = risk_manager.size_position(context)

    assert advice.target_notional == Decimal("5000")
    assert captured["context"].symbol == "BTC-PERP"

    events = [
        m for m in store.metrics if m["metrics"].get("event_type") == "position_sizing_advice"
    ]
    assert events and events[-1]["metrics"]["symbol"] == "BTC-PERP"


def test_size_position_fallback_on_failure():
    config = RiskConfig(enable_dynamic_position_sizing=True)
    store = RecordingEventStore()

    def estimator(context: PositionSizingContext) -> PositionSizingAdvice:
        raise RuntimeError("estimator blew up")

    risk_manager = LiveRiskManager(
        config=config,
        event_store=store,
        position_size_estimator=estimator,
    )

    context = _make_context()
    advice = risk_manager.size_position(context)

    assert advice.fallback_used is True
    assert advice.target_notional > 0

    error_events = [
        m for m in store.metrics if m["metrics"].get("event_type") == "position_sizing_error"
    ]
    assert error_events


def test_size_position_respects_reduce_only_mode():
    config = RiskConfig(enable_dynamic_position_sizing=True)
    store = RecordingEventStore()

    def estimator(context: PositionSizingContext) -> PositionSizingAdvice:
        return PositionSizingAdvice(
            symbol=context.symbol,
            side=context.side,
            target_notional=Decimal("6000"),
            target_quantity=Decimal("0.12"),
            used_dynamic=True,
            reduce_only=False,
        )

    risk_manager = LiveRiskManager(
        config=config,
        event_store=store,
        position_size_estimator=estimator,
    )

    risk_manager.set_reduce_only_mode(True, reason="test")
    context = _make_context()
    advice = risk_manager.size_position(context)

    assert advice.target_notional == Decimal("0")
    assert advice.reduce_only is True

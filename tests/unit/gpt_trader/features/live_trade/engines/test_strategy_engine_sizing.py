"""Tests for TradingEngine sizing and quantity calculations."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Product
from gpt_trader.features.live_trade.execution.submission_result import OrderSubmissionStatus
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, Decision


@pytest.mark.asyncio
async def test_quantity_zero_emits_gate_event(engine) -> None:
    result = await engine._validate_and_place_order(
        symbol="BTC-USD",
        decision=Decision(Action.BUY, "test"),
        price=Decimal("50000"),
        equity=Decimal("10000"),
        quantity_override=Decimal("0"),
    )

    assert result.status == OrderSubmissionStatus.BLOCKED
    events = [
        event
        for event in engine._event_store.list_events()
        if event.get("type") == "trade_gate_blocked"
    ]
    assert events
    payload = events[-1].get("data", {})
    assert payload.get("gate") == "sizing"
    assert payload.get("reason") == "quantity_zero"


def test_calculate_order_quantity_with_strategy_config(engine):
    """Test quantity calculation uses strategy config if set."""
    engine.strategy.config.position_fraction = Decimal("0.5")
    equity = Decimal("10000")
    price = Decimal("50000")

    quantity = engine._calculate_order_quantity("BTC-USD", price, equity, None)
    assert quantity == Decimal("0.1")


def test_calculate_order_quantity_fallback_to_bot_config(engine):
    """Test quantity calculation falls back to bot config."""
    engine.strategy.config.position_fraction = None
    engine.context.config.perps_position_fraction = Decimal("0.2")

    equity = Decimal("10000")
    price = Decimal("50000")

    quantity = engine._calculate_order_quantity("BTC-USD", price, equity, None)
    assert quantity == Decimal("0.04")


def test_calculate_order_quantity_min_size(engine):
    """Test quantity respects product min size."""
    engine.strategy.config.position_fraction = Decimal("0.1")
    equity = Decimal("100")
    price = Decimal("50000")

    product = MagicMock(spec=Product)
    product.min_size = Decimal("0.001")

    quantity = engine._calculate_order_quantity("BTC-USD", price, equity, product)
    assert quantity == Decimal("0")

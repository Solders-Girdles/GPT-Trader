"""Tests for position snapshot and metric recording."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from tests.unit.bot_v2.orchestration.helpers import ScenarioBuilder

from bot_v2.features.brokerages.core.interfaces import OrderStatus


@pytest.mark.asyncio
async def test_snapshot_positions_returns_normalized_map(reconciler, fake_broker):
    valid_position = SimpleNamespace(
        symbol="BTC-PERP",
        quantity=Decimal("1.5"),
        side="long",
    )
    missing_symbol = SimpleNamespace(
        symbol=None,
        quantity=Decimal("2"),
        side="short",
    )
    invalid_quantity = SimpleNamespace(
        symbol="ETH-PERP",
        quantity=None,
        side="long",
    )
    fake_broker.list_positions.return_value = [valid_position, missing_symbol, invalid_quantity]

    snapshot = await reconciler.snapshot_positions()

    assert snapshot == {
        "BTC-PERP": {"quantity": "1.5", "side": "long"},
    }


@pytest.mark.asyncio
async def test_snapshot_positions_handles_broker_exception(reconciler, fake_broker):
    fake_broker.list_positions.side_effect = RuntimeError("fail")

    snapshot = await reconciler.snapshot_positions()

    assert snapshot == {}


@pytest.mark.asyncio
async def test_record_snapshot_emits_metric(monkeypatch, reconciler, fake_event_store):
    captured: list[dict[str, Any]] = []

    def fake_emit(store, bot_id, payload, logger):
        captured.append({"store": store, "bot_id": bot_id, "payload": payload})

    monkeypatch.setattr(
        "bot_v2.orchestration.order_reconciler.emit_metric",
        fake_emit,
    )

    local = {"one": ScenarioBuilder.create_order(id="local1", status=OrderStatus.SUBMITTED)}
    exchange = {
        "two": ScenarioBuilder.create_order(id="ex1", status=OrderStatus.SUBMITTED),
        "three": ScenarioBuilder.create_order(id="ex2", status=OrderStatus.PENDING),
    }

    await reconciler.record_snapshot(local, exchange)

    assert captured == [
        {
            "store": fake_event_store,
            "bot_id": "test-bot",
            "payload": {
                "event_type": "order_reconcile_snapshot",
                "local_open": 1,
                "exchange_open": 2,
            },
        }
    ]


@pytest.mark.asyncio
async def test_record_snapshot_with_empty_order_dicts(reconciler, fake_event_store, monkeypatch):
    """Test record_snapshot edge case with empty order dictionaries."""
    captured: list[dict[str, Any]] = []

    def fake_emit(store, bot_id, payload, logger):
        captured.append({"store": store, "bot_id": bot_id, "payload": payload})

    monkeypatch.setattr(
        "bot_v2.orchestration.order_reconciler.emit_metric",
        fake_emit,
    )

    # Test with empty order dictionaries
    local = {}
    exchange = {}

    await reconciler.record_snapshot(local, exchange)

    assert captured == [
        {
            "store": fake_event_store,
            "bot_id": "test-bot",
            "payload": {
                "event_type": "order_reconcile_snapshot",
                "local_open": 0,
                "exchange_open": 0,
            },
        }
    ]


@pytest.mark.asyncio
async def test_record_snapshot_handles_metric_emission_failure(reconciler, fake_event_store, monkeypatch):
    """Test record_snapshot when metric emission fails."""
    local = {"one": ScenarioBuilder.create_order(id="local1", status=OrderStatus.SUBMITTED)}
    exchange = {"two": ScenarioBuilder.create_order(id="ex1", status=OrderStatus.SUBMITTED)}

    # Mock metric emission to fail
    def failing_emit(*args, **kwargs):
        raise RuntimeError("Metric emission failed")

    monkeypatch.setattr("bot_v2.orchestration.order_reconciler.emit_metric", failing_emit)

    # The actual implementation doesn't handle metric emission failure gracefully,
    # so we expect it to raise an exception
    with pytest.raises(RuntimeError, match="Metric emission failed"):
        await reconciler.record_snapshot(local, exchange)



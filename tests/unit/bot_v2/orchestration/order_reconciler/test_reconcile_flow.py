"""Reconciliation flow tests for OrderReconciler."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.unit.bot_v2.orchestration.helpers import ScenarioBuilder

from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from bot_v2.orchestration.order_reconciler import OrderDiff, OrderReconciler


@pytest.mark.asyncio
async def test_reconcile_missing_on_exchange_persists_final_order(
    reconciler, fake_broker, monkeypatch
) -> None:
    diff = OrderDiff(
        missing_on_exchange={"missing": ScenarioBuilder.create_order(id="missing")},
        missing_locally={},
    )
    final_order = Order(
        id="missing",
        client_id=None,
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        tif=TimeInForce.GTC,
        status=OrderStatus.FILLED,
        submitted_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        quantity=Decimal("1"),
        filled_quantity=Decimal("1"),
    )
    fake_broker.get_order.return_value = final_order
    persist = Mock()
    assume = Mock()
    monkeypatch.setattr(reconciler, "_persist_exchange_update", persist)
    monkeypatch.setattr(reconciler, "_assume_cancelled", assume)

    await reconciler.reconcile_missing_on_exchange(diff)

    persist.assert_called_once_with(final_order)
    assume.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_missing_on_exchange_assumes_cancelled_when_missing(
    reconciler, fake_broker, monkeypatch
) -> None:
    diff = OrderDiff(
        missing_on_exchange={"missing": ScenarioBuilder.create_order(id="missing")},
        missing_locally={},
    )
    fake_broker.get_order.return_value = None
    assume = Mock()
    monkeypatch.setattr(reconciler, "_assume_cancelled", assume)

    await reconciler.reconcile_missing_on_exchange(diff)

    assume.assert_called_once()


@pytest.mark.asyncio
async def test_reconcile_missing_on_exchange_handles_broker_exception(
    reconciler, fake_broker, monkeypatch
) -> None:
    diff = OrderDiff(
        missing_on_exchange={"missing": ScenarioBuilder.create_order(id="missing")},
        missing_locally={},
    )
    fake_broker.get_order.side_effect = RuntimeError("boom")
    assume = Mock()
    monkeypatch.setattr(reconciler, "_assume_cancelled", assume)

    await reconciler.reconcile_missing_on_exchange(diff)

    assume.assert_called_once()


def test_reconcile_missing_locally_upserts_orders(reconciler, fake_orders_store):
    exchange_order = ScenarioBuilder.create_order(id="exchange-order", status=OrderStatus.SUBMITTED)
    diff = OrderDiff(missing_on_exchange={}, missing_locally={"exchange": exchange_order})

    reconciler.reconcile_missing_locally(diff)

    fake_orders_store.upsert.assert_called_once_with(exchange_order)


def test_reconcile_missing_locally_handles_upsert_error(reconciler, fake_orders_store):
    exchange_order = ScenarioBuilder.create_order(id="exchange-order", status=OrderStatus.SUBMITTED)
    diff = OrderDiff(missing_on_exchange={}, missing_locally={"exchange": exchange_order})
    fake_orders_store.upsert.side_effect = RuntimeError("fail")

    reconciler.reconcile_missing_locally(diff)

    assert fake_orders_store.upsert.call_count == 1


def _build_local_order() -> SimpleNamespace:
    return SimpleNamespace(
        order_id="local-1",
        client_id="client-123",
        symbol="BTC-PERP",
        side="sell",
        order_type="limit",
        quantity=Decimal("2"),
        price="100.50",
        avg_fill_price="99.5",
        filled_quantity=Decimal("1.5"),
        created_at="2024-05-31T10:00:00",
    )


def test_assume_cancelled_creates_cancelled_order(
    reconciler, fake_orders_store, fake_event_store, monkeypatch
) -> None:
    emitted = []

    def fake_emit(store, bot_id, payload, logger):
        emitted.append(payload)

    monkeypatch.setattr("bot_v2.orchestration.order_reconciler.emit_metric", fake_emit)

    local_order = _build_local_order()

    reconciler._assume_cancelled("missing-order", local_order)

    upserted = fake_orders_store.upsert.call_args[0][0]
    assert upserted.id == "local-1"
    assert upserted.status == OrderStatus.CANCELLED
    assert upserted.side == OrderSide.SELL
    assert upserted.type == OrderType.LIMIT
    assert upserted.quantity == Decimal("2")
    assert upserted.price == Decimal("100.50")
    assert upserted.avg_fill_price == Decimal("99.5")
    assert upserted.filled_quantity == Decimal("1.5")
    assert emitted == [
        {
            "event_type": "order_reconciled",
            "order_id": "missing-order",
            "status": OrderStatus.CANCELLED.value,
            "reason": "assumed_cancelled",
        }
    ]


def test_assume_cancelled_handles_upsert_failure(reconciler, fake_orders_store, monkeypatch):
    fake_orders_store.upsert.side_effect = RuntimeError("fail")
    monkeypatch.setattr(
        "bot_v2.orchestration.order_reconciler.emit_metric",
        lambda *args, **kwargs: None,
    )

    reconciler._assume_cancelled("missing-order", _build_local_order())

    assert fake_orders_store.upsert.call_count == 1


def test_parse_timestamp_with_valid_value():
    ts = "2024-05-31T10:00:00"
    parsed = OrderReconciler._parse_timestamp(ts)
    assert parsed == datetime.fromisoformat(ts)


def test_parse_timestamp_invalid_returns_none():
    parsed = OrderReconciler._parse_timestamp("not-a-date")
    assert parsed is None


def test_persist_exchange_update_handles_upsert_error(reconciler, fake_orders_store, fake_event_store, monkeypatch):
    """Test _persist_exchange_update error handling when upsert fails."""
    from bot_v2.features.brokerages.core.interfaces import OrderStatus, OrderSide, OrderType, TimeInForce
    from datetime import UTC, datetime
    from decimal import Decimal

    # Create a test order
    test_order = Order(
        id="test-order",
        client_id=None,
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        tif=TimeInForce.GTC,
        status=OrderStatus.FILLED,
        submitted_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        quantity=Decimal("1"),
        filled_quantity=Decimal("1"),
    )

    # Mock upsert to raise an exception
    fake_orders_store.upsert.side_effect = RuntimeError("Database connection failed")

    # Mock metric emission to prevent errors during cleanup
    def mock_emit(*args, **kwargs):
        pass
    monkeypatch.setattr("bot_v2.orchestration.order_reconciler.emit_metric", mock_emit)

    # Should not raise exception despite upsert failure
    result = reconciler._persist_exchange_update(test_order)

    # Should return None when upsert fails (error handling path)
    assert result is None
    # Verify upsert was attempted
    fake_orders_store.upsert.assert_called_once_with(test_order)


def test_assume_cancelled_handles_outer_exception(reconciler, fake_orders_store, fake_event_store, monkeypatch):
    """Test _assume_cancelled outer exception handling when upsert and metric emission fail."""
    from bot_v2.features.brokerages.core.interfaces import OrderStatus

    # Mock upsert to fail
    fake_orders_store.upsert.side_effect = RuntimeError("Upsert failed")

    # Mock metric emission to also fail
    def failing_emit(*args, **kwargs):
        raise RuntimeError("Metric emission failed")
    monkeypatch.setattr("bot_v2.orchestration.order_reconciler.emit_metric", failing_emit)

    # Should handle both failures gracefully without crashing
    # This tests the outer try-catch block in _assume_cancelled
    reconciler._assume_cancelled("test-order", _build_local_order())

    # Verify upsert was attempted
    assert fake_orders_store.upsert.call_count == 1


def test_assume_cancelled_handles_malformed_order_data(reconciler, fake_orders_store, fake_event_store, monkeypatch):
    """Test _assume_cancelled with malformed order data."""
    # Create malformed order data with missing/invalid fields
    malformed_order = SimpleNamespace(
        order_id="malformed-1",
        # Missing required fields like client_id, symbol, side, etc.
        created_at="invalid-date",  # Invalid timestamp
        quantity="not-a-decimal",  # Invalid quantity format
    )

    # Mock metric emission to prevent errors
    def mock_emit(*args, **kwargs):
        pass
    monkeypatch.setattr("bot_v2.orchestration.order_reconciler.emit_metric", mock_emit)

    # Should handle malformed order without crashing
    reconciler._assume_cancelled("malformed-order", malformed_order)

    # Verify upsert was still attempted (even with malformed data)
    fake_orders_store.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_reconcile_missing_on_exchange_handles_complex_order_data(reconciler, fake_broker, monkeypatch):
    """Test reconciliation with complex/malformed order data from broker."""
    diff = OrderDiff(
        missing_on_exchange={"complex": ScenarioBuilder.create_order(id="complex")},
        missing_locally={},
    )

    # Mock broker to return order with missing fields
    complex_order = Mock()
    complex_order.id = "complex"
    complex_order.status = "filled"  # String instead of enum
    # Missing other required fields

    fake_broker.get_order.return_value = complex_order

    persist = Mock()
    monkeypatch.setattr(reconciler, "_persist_exchange_update", persist)

    # Should handle complex order data without crashing
    await reconciler.reconcile_missing_on_exchange(diff)

    # Should still attempt to persist the order
    persist.assert_called_once_with(complex_order)


def test_reconcile_missing_locally_handles_order_validation_errors(reconciler, fake_orders_store):
    """Test reconciliation when exchange order has validation errors."""
    # Create an exchange order that will fail validation/upsert
    problematic_order = Mock()
    problematic_order.id = "problematic"
    # Missing or invalid required fields

    diff = OrderDiff(missing_on_exchange={}, missing_locally={"problematic": problematic_order})

    # Mock upsert to simulate validation failure
    fake_orders_store.upsert.side_effect = ValueError("Invalid order data")

    # Should handle validation errors gracefully
    reconciler.reconcile_missing_locally(diff)

    # Verify upsert was attempted
    assert fake_orders_store.upsert.call_count == 1

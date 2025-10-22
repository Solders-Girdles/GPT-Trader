"""Tests for fetching and diffing orders in OrderReconciler."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from tests.unit.bot_v2.orchestration.helpers import ScenarioBuilder

from bot_v2.features.brokerages.core.interfaces import OrderStatus
from bot_v2.orchestration.order_reconciler import OrderDiff, OrderReconciler


def test_fetch_local_open_orders_returns_mapping(reconciler, fake_orders_store):
    local_order = ScenarioBuilder.create_order(id="order-1")
    local_order.order_id = "order-1"
    fake_orders_store.get_open_orders.return_value = [local_order]

    result = reconciler.fetch_local_open_orders()

    assert result == {"order-1": local_order}


def test_fetch_local_open_orders_handles_exception(reconciler, fake_orders_store):
    fake_orders_store.get_open_orders.side_effect = RuntimeError("boom")

    result = reconciler.fetch_local_open_orders()

    assert result == {}


@pytest.mark.asyncio
async def test_fetch_exchange_open_orders_collects_statuses(reconciler, fake_broker):
    orders_by_status = {
        OrderStatus.PENDING: [
            ScenarioBuilder.create_order(id="pending-1", status=OrderStatus.PENDING)
        ],
        OrderStatus.SUBMITTED: [
            ScenarioBuilder.create_order(id="submitted-1", status=OrderStatus.SUBMITTED)
        ],
        OrderStatus.PARTIALLY_FILLED: [
            ScenarioBuilder.create_order(id="partial-1", status=OrderStatus.PARTIALLY_FILLED)
        ],
    }

    def list_orders(status=None):
        return orders_by_status.get(status, [])

    fake_broker.list_orders.side_effect = list_orders

    result = await reconciler.fetch_exchange_open_orders()

    assert set(result.keys()) == {"pending-1", "submitted-1", "partial-1"}


@pytest.mark.asyncio
async def test_fetch_exchange_open_orders_fallback_on_type_error(reconciler, fake_broker):
    fallback_orders = [
        ScenarioBuilder.create_order(id="fallback-1", status=OrderStatus.SUBMITTED),
        ScenarioBuilder.create_order(id="fallback-2", status=OrderStatus.CANCELLED),
    ]

    def list_orders(status=None):
        if status is None:
            return fallback_orders
        raise TypeError("list_orders does not support status arg")

    fake_broker.list_orders.side_effect = list_orders

    result = await reconciler.fetch_exchange_open_orders()

    assert "fallback-1" in result
    assert "fallback-2" not in result  # filtered by interested statuses


@pytest.mark.asyncio
async def test_fetch_exchange_open_orders_handles_exceptions(reconciler, fake_broker):
    calls = 0

    def list_orders(status=None):
        nonlocal calls
        calls += 1
        if status == OrderStatus.PENDING:
            raise RuntimeError("broker down")
        return [ScenarioBuilder.create_order(id=f"order-{calls}", status=status)]

    fake_broker.list_orders.side_effect = list_orders

    result = await reconciler.fetch_exchange_open_orders()

    assert "order-2" in result  # from submitted status despite error on pending


def test_diff_orders_identifies_missing_orders():
    local = {
        "local-only": ScenarioBuilder.create_order(id="local-only"),
        "shared": ScenarioBuilder.create_order(id="shared"),
    }
    exchange = {
        "exchange-only": ScenarioBuilder.create_order(id="exchange-only"),
        "shared": ScenarioBuilder.create_order(id="shared"),
    }

    diff = OrderDiff(
        missing_on_exchange={"local-only": local["local-only"]},
        missing_locally={"exchange-only": exchange["exchange-only"]},
    )

    computed = OrderReconciler.diff_orders(local, exchange)

    assert computed == diff


@pytest.mark.asyncio
async def test_fetch_exchange_open_orders_fallback_success_filters_status(reconciler, fake_broker):
    """Test fallback success path when broker doesn't support status parameter."""
    reconciler_instance = reconciler  # Use the fixture
    fake_broker_instance = fake_broker  # Use the fixture

    # Mock broker that doesn't support status parameter but returns all orders
    all_orders = [
        ScenarioBuilder.create_order(id="pending-order", status=OrderStatus.PENDING),
        ScenarioBuilder.create_order(id="submitted-order", status=OrderStatus.SUBMITTED),
        ScenarioBuilder.create_order(id="partial-order", status=OrderStatus.PARTIALLY_FILLED),
        ScenarioBuilder.create_order(
            id="filled-order", status=OrderStatus.FILLED
        ),  # Should be filtered out
        ScenarioBuilder.create_order(
            id="cancelled-order", status=OrderStatus.CANCELLED
        ),  # Should be filtered out
    ]

    def list_orders(status=None):
        if status is None:
            # Return all orders when called without status (fallback mode)
            return all_orders
        raise TypeError("list_orders does not support status arg")

    fake_broker_instance.list_orders.side_effect = list_orders

    result = await reconciler_instance.fetch_exchange_open_orders()

    # Should include only orders with interested statuses (not filled/cancelled)
    assert "pending-order" in result
    assert "submitted-order" in result
    assert "partial-order" in result
    assert "filled-order" not in result  # Filled orders are not "open"
    assert "cancelled-order" not in result  # Cancelled orders are not "open"


@pytest.mark.asyncio
async def test_fetch_exchange_open_orders_fallback_with_empty_orders(reconciler, fake_broker):
    """Test fallback when broker returns empty order list."""
    reconciler_instance = reconciler  # Use the fixture
    fake_broker_instance = fake_broker  # Use the fixture

    def list_orders(status=None):
        if status is None:
            return []  # Empty orders in fallback mode
        raise TypeError("list_orders does not support status arg")

    fake_broker_instance.list_orders.side_effect = list_orders

    result = await reconciler_instance.fetch_exchange_open_orders()

    # Should return empty dict when no orders are returned
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_exchange_open_orders_fallback_with_malformed_orders(reconciler, fake_broker):
    """Test fallback when broker returns orders with missing required fields."""
    reconciler_instance = reconciler  # Use the fixture
    fake_broker_instance = fake_broker  # Use the fixture

    # Create orders with missing/invalid fields
    malformed_order = Mock()
    malformed_order.id = "malformed"
    # Missing other required fields like status, symbol, etc.

    def list_orders(status=None):
        if status is None:
            return [malformed_order]
        raise TypeError("list_orders does not support status arg")

    fake_broker_instance.list_orders.side_effect = list_orders

    # Should handle malformed orders gracefully without crashing
    result = await reconciler_instance.fetch_exchange_open_orders()
    # The exact behavior depends on implementation - test that it doesn't crash
    assert isinstance(result, dict)

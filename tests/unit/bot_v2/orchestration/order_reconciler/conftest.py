"""Shared fixtures for order_reconciler tests."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, Mock

import pytest

from bot_v2.orchestration.order_reconciler import OrderReconciler
from bot_v2.persistence.event_store import EventStore
from bot_v2.persistence.orders_store import OrdersStore


@pytest.fixture(autouse=True)
def patch_async_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run asyncio.to_thread synchronously for deterministic tests."""

    async def _immediate(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _immediate)


@pytest.fixture
def fake_orders_store() -> MagicMock:
    store = MagicMock(spec=OrdersStore)
    store.get_open_orders.return_value = []
    return store


@pytest.fixture
def fake_event_store() -> MagicMock:
    return MagicMock(spec=EventStore)


@pytest.fixture
def fake_broker() -> Mock:
    broker = Mock()
    broker.list_orders = Mock(return_value=[])
    broker.get_order = Mock()
    broker.list_positions = Mock(return_value=[])
    return broker


@pytest.fixture
def reconciler(
    fake_broker: Mock,
    fake_orders_store: MagicMock,
    fake_event_store: MagicMock,
) -> OrderReconciler:
    return OrderReconciler(
        broker=fake_broker,
        orders_store=fake_orders_store,
        event_store=fake_event_store,
        bot_id="test-bot",
    )

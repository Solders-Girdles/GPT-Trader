"""Tests for emergency flatten close-order audit records."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

import gpt_trader.features.live_trade.bot as bot_module
from gpt_trader.app.config import BotConfig
from gpt_trader.features.live_trade.bot import TradingBot


class _DirectBrokerCalls:
    def __init__(self) -> None:
        self.shutdown = Mock()

    async def __call__(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _install_mock_engine(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    engine = AsyncMock()
    engine.shutdown = AsyncMock()
    engine.preserve_broker_calls_on_shutdown = MagicMock()
    monkeypatch.setattr(bot_module, "TradingEngine", MagicMock(return_value=engine))
    return engine


@pytest.mark.asyncio
async def test_flatten_and_stop_persists_submitted_close_order_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gpt_trader.core import OrderSide, OrderType

    broker = Mock()
    broker.list_positions.return_value = [SimpleNamespace(symbol="BTC-USD", quantity=Decimal("1"))]
    broker.place_order = Mock(
        return_value={
            "order_id": "close-btc-123",
            "client_order_id": "client-close-btc-123",
            "status": "FILLED",
            "filled_size": "1",
            "average_filled_price": "50000.12",
        }
    )
    event_store = Mock()
    orders_store = Mock()
    container = SimpleNamespace(
        broker=broker,
        risk_manager=Mock(),
        event_store=event_store,
        orders_store=orders_store,
        notification_service=Mock(),
    )
    engine = _install_mock_engine(monkeypatch)

    bot = TradingBot(
        config=BotConfig(symbols=["BTC-USD"], interval=1),
        container=container,
    )
    bot._broker_calls = _DirectBrokerCalls()

    messages = await bot.flatten_and_stop()

    assert messages[-1] == "Bot stopped."
    broker.place_order.assert_called_once_with(
        symbol="BTC-USD",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("1"),
        reduce_only=True,
    )
    engine.shutdown.assert_called_once()
    close_order_events = [
        call.args[1]
        for call in event_store.append.call_args_list
        if call.args[0] == "emergency_flatten_close_order"
    ]
    assert len(close_order_events) == 1
    audit_payload = close_order_events[0]
    assert audit_payload["flatten_operation_id"].startswith("flatten-")
    assert audit_payload["symbol"] == "BTC-USD"
    assert audit_payload["side"] == "SELL"
    assert audit_payload["order_type"] == "MARKET"
    assert audit_payload["quantity"] == "1"
    assert audit_payload["reduce_only"] is True
    assert audit_payload["status"] == "submitted"
    assert audit_payload["broker_status"] == "FILLED"
    assert audit_payload["filled_quantity"] == "1"
    assert audit_payload["average_fill_price"] == "50000.12"
    assert audit_payload["order_id"] == "close-btc-123"
    assert audit_payload["client_order_id"] == "client-close-btc-123"

    orders_store.upsert_by_client_id.assert_called_once()
    order_record = orders_store.upsert_by_client_id.call_args.args[0]
    assert order_record.order_id == "close-btc-123"
    assert order_record.client_order_id == "client-close-btc-123"
    assert order_record.symbol == "BTC-USD"
    assert order_record.side == "sell"
    assert order_record.order_type == "market"
    assert order_record.status.value == "filled"
    assert order_record.filled_quantity == Decimal("1")
    assert order_record.average_fill_price == Decimal("50000.12")
    assert order_record.metadata["source"] == "emergency_flatten"
    assert order_record.metadata["flatten_operation_id"] == audit_payload["flatten_operation_id"]


@pytest.mark.asyncio
async def test_flatten_and_stop_links_partial_failure_audit_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def place_order(**kwargs):
        if kwargs["symbol"] == "ETH-USD":
            raise RuntimeError("venue rejected emergency close")
        return SimpleNamespace(
            id="close-btc-123",
            client_order_id="client-close-btc-123",
            status="OPEN",
        )

    broker = Mock()
    broker.list_positions.return_value = [
        SimpleNamespace(symbol="BTC-USD", quantity=Decimal("1")),
        SimpleNamespace(symbol="ETH-USD", quantity=Decimal("-2")),
    ]
    broker.place_order = Mock(side_effect=place_order)
    event_store = Mock()
    orders_store = Mock()
    notification_service = Mock()
    notification_service.notify = AsyncMock(return_value=True)
    container = SimpleNamespace(
        broker=broker,
        risk_manager=Mock(),
        event_store=event_store,
        orders_store=orders_store,
        notification_service=notification_service,
    )
    _install_mock_engine(monkeypatch)

    bot = TradingBot(
        config=BotConfig(symbols=["BTC-USD", "ETH-USD"], interval=1),
        container=container,
    )
    bot._broker_calls = _DirectBrokerCalls()

    messages = await bot.flatten_and_stop()

    assert any("Emergency flatten incomplete" in msg for msg in messages)
    close_order_events = [
        call.args[1]
        for call in event_store.append.call_args_list
        if call.args[0] == "emergency_flatten_close_order"
    ]
    assert len(close_order_events) == 2
    success_event = next(event for event in close_order_events if event["symbol"] == "BTC-USD")
    failed_event = next(event for event in close_order_events if event["symbol"] == "ETH-USD")
    flatten_operation_id = success_event["flatten_operation_id"]
    assert failed_event["flatten_operation_id"] == flatten_operation_id
    assert success_event["order_id"] == "close-btc-123"
    assert success_event["client_order_id"] == "client-close-btc-123"
    assert success_event["status"] == "submitted"
    assert failed_event["status"] == "failed"
    assert failed_event["error"] == "venue rejected emergency close"

    orders_store.upsert_by_client_id.assert_called_once()
    order_record = orders_store.upsert_by_client_id.call_args.args[0]
    assert order_record.metadata["flatten_operation_id"] == flatten_operation_id

    flatten_failure_events = [
        call.args[1]
        for call in event_store.append.call_args_list
        if call.args[0] == "emergency_flatten_failed"
    ]
    assert len(flatten_failure_events) == 1
    payload = flatten_failure_events[0]
    assert payload["flatten_operation_id"] == flatten_operation_id
    assert payload["failed_closes"][0]["flatten_operation_id"] == flatten_operation_id


@pytest.mark.asyncio
async def test_flatten_and_stop_avoids_filled_record_without_fill_quantity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = Mock()
    broker.list_positions.return_value = [SimpleNamespace(symbol="BTC-USD", quantity=Decimal("1"))]
    broker.place_order = Mock(
        return_value={
            "order_id": "close-btc-123",
            "client_order_id": "client-close-btc-123",
            "status": "FILLED",
        }
    )
    event_store = Mock()
    orders_store = Mock()
    container = SimpleNamespace(
        broker=broker,
        risk_manager=Mock(),
        event_store=event_store,
        orders_store=orders_store,
        notification_service=Mock(),
    )
    _install_mock_engine(monkeypatch)

    bot = TradingBot(
        config=BotConfig(symbols=["BTC-USD"], interval=1),
        container=container,
    )
    bot._broker_calls = _DirectBrokerCalls()

    await bot.flatten_and_stop()

    orders_store.upsert_by_client_id.assert_called_once()
    order_record = orders_store.upsert_by_client_id.call_args.args[0]
    assert order_record.status.value == "open"
    assert order_record.filled_quantity == Decimal("0")
    assert order_record.average_fill_price is None
    audit_payload = next(
        call.args[1]
        for call in event_store.append.call_args_list
        if call.args[0] == "emergency_flatten_close_order"
    )
    assert audit_payload["broker_status"] == "FILLED"
    assert "filled_quantity" not in audit_payload
    assert "average_fill_price" not in audit_payload

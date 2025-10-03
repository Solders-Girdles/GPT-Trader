from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from bot_v2.errors import ValidationError
from bot_v2.features.brokerages.coinbase.rest import orders as orders_module
from bot_v2.features.brokerages.core.interfaces import (
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
)


class DummyEndpoints:
    def list_orders(self) -> str:
        return "/orders"

    def get_order(self, order_id: str) -> str:
        return f"/orders/{order_id}"


class DummyClient:
    def __init__(self) -> None:
        self.preview_payloads: list[dict[str, Any]] = []
        self.edit_previews: list[dict[str, Any]] = []
        self.edit_orders: list[dict[str, Any]] = []
        self.place_calls: list[dict[str, Any]] = []
        self.cancelled: list[list[str]] = []
        self.list_params: list[dict[str, str]] = []
        self.close_payloads: list[dict[str, Any]] = []

    def preview_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.preview_payloads.append(payload)
        return {"ok": True, "payload": payload}

    def edit_order_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.edit_previews.append(payload)
        return {"preview": payload}

    def edit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.edit_orders.append(payload)
        return {"order": payload}

    def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.place_calls.append(payload)
        return {"id": "placed", **payload}

    def cancel_orders(self, order_ids: list[str]) -> dict[str, Any]:
        self.cancelled.append(order_ids)
        return {"results": [{"order_id": order_ids[0], "success": True}]}

    def list_orders(self, **params: str) -> dict[str, Any]:
        self.list_params.append(params)
        return {"orders": [{"id": "o-1"}, {"id": "o-2"}]}

    def get_order_historical(self, order_id: str) -> dict[str, Any]:
        return {"order": {"id": order_id}}

    def list_fills(self, **params: str) -> dict[str, Any]:
        return {"fills": [{"id": "fill-1"}]}

    def close_position(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.close_payloads.append(payload)
        return {"order": {"closed": True, **payload}}


class DummyOrderService(orders_module.OrderRestMixin):
    def __init__(self) -> None:
        self.client = DummyClient()
        self.endpoints = DummyEndpoints()
        self._positions = [
            Position(
                symbol="BTC_USD",
                quantity=Decimal("1"),
                entry_price=Decimal("100"),
                mark_price=Decimal("105"),
                unrealized_pnl=Decimal("5"),
                realized_pnl=Decimal("0"),
                leverage=5,
                side="long",
            )
        ]

    def _build_order_payload(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        return kwargs

    def _execute_order_payload(self, symbol: str, payload: dict[str, Any], client_id: str | None) -> dict[str, Any]:  # type: ignore[override]
        return {"symbol": symbol, "payload": payload, "client_id": client_id}

    def list_positions(self) -> list[Position]:  # type: ignore[override]
        return self._positions


@pytest.fixture(autouse=True)
def _patch_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orders_module, "normalize_symbol", lambda symbol: symbol.replace("-", "_"))
    monkeypatch.setattr(orders_module, "to_order", lambda payload: payload)


def test_require_quantity_validation() -> None:
    with pytest.raises(ValueError):
        orders_module.OrderRestMixin._require_quantity(None, context="test")

    assert orders_module.OrderRestMixin._require_quantity(Decimal("1"), context="ctx") == Decimal(
        "1"
    )


def test_preview_and_edit_flows() -> None:
    service = DummyOrderService()
    response = service.preview_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1"),
    )
    assert response["ok"]
    assert service.client.preview_payloads

    edit_preview = service.edit_order_preview(
        order_id="abc",
        symbol="BTC-USD",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal("2"),
        price=Decimal("99"),
    )
    assert edit_preview["preview"]["price"] == Decimal("99")

    edited = service.edit_order("abc", "preview-1")
    assert edited["order"]["order_id"] == "abc"


def test_place_and_cancel_order() -> None:
    service = DummyOrderService()
    placed = service.place_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.5"),
        tif=TimeInForce.GTC,
    )
    assert placed["payload"]["quantity"] == Decimal("0.5")

    assert service.cancel_order("ord-1")
    assert service.client.cancelled[0] == ["ord-1"]


def test_list_orders_and_get_order(monkeypatch: pytest.MonkeyPatch) -> None:
    service = DummyOrderService()
    orders = service.list_orders(status=OrderStatus.SUBMITTED, symbol="BTC-USD")
    assert len(orders) == 2
    assert service.client.list_params[0]["product_id"] == "BTC_USD"

    retrieved = service.get_order("ord-1")
    assert retrieved["id"] == "ord-1"

    def broken(*_: Any, **__: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(service.client, "list_orders", broken)
    assert service.list_orders() == []

    monkeypatch.setattr(service.client, "get_order_historical", broken)
    assert service.get_order("missing") is None


def test_list_fills_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    service = DummyOrderService()
    fills = service.list_fills(symbol="BTC-USD", limit=50)
    assert fills[0]["id"] == "fill-1"

    monkeypatch.setattr(
        service.client, "list_fills", lambda **_: (_ for _ in ()).throw(RuntimeError("fail"))
    )
    assert service.list_fills() == []


def test_close_position_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    service = DummyOrderService()
    order = service.close_position("BTC-USD")
    assert order["closed"]
    assert service.client.close_payloads[0]["side"] == "SELL"

    # Provide explicit quantity path
    service._positions = []
    fallback_called: dict[str, Any] = {}

    def fallback(side: OrderSide, qty: Decimal, reduce_only: bool) -> dict[str, Any]:
        fallback_called.update({"side": side, "qty": qty, "reduce_only": reduce_only})
        return {"fallback": True}

    monkeypatch.setattr(
        service.client,
        "close_position",
        lambda payload: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    order = service.close_position(
        "ETH-USD",
        quantity=Decimal("3"),
        positions_override=[
            Position(
                symbol="ETH_USD",
                quantity=Decimal("-3"),
                entry_price=Decimal("200"),
                mark_price=Decimal("198"),
                unrealized_pnl=Decimal("-6"),
                realized_pnl=Decimal("0"),
                leverage=4,
                side="short",
            )
        ],
        fallback=fallback,
    )
    assert order["fallback"]
    assert fallback_called["side"] is OrderSide.BUY

    # No position scenario should raise
    with pytest.raises(ValidationError):
        service.close_position("LTC-USD", positions_override=[])

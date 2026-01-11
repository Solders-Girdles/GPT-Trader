from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from gpt_trader.features.brokerages.coinbase.client.orders import OrderClientMixin
from gpt_trader.features.brokerages.coinbase.errors import (
    BrokerageError,
    InvalidRequestError,
    NotFoundError,
)


class _StubOrderClient(OrderClientMixin):
    def __init__(self, api_mode: str = "advanced", responses: list[Any] | None = None) -> None:
        self.api_mode = api_mode
        self._responses = list(responses or [])
        self.calls: list[tuple[str, str, Any | None]] = []

    def _get_endpoint_path(self, endpoint_name: str, **kwargs: str) -> str:
        if endpoint_name == "order":
            return f"/orders/{kwargs['order_id']}"
        return f"/{endpoint_name}"

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        self.calls.append((method, path, payload))
        if self._responses:
            response = self._responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        return {"ok": True}


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs"),
    [
        ("place_order", ({"product_id": "BTC-USD"},), {}),
        ("preview_order", ({"product_id": "BTC-USD"},), {}),
        ("edit_order_preview", ({"order_id": "order-1"},), {}),
        ("edit_order", ({"order_id": "order-1"},), {}),
        ("close_position", ({"product_id": "BTC-USD"},), {}),
        ("list_orders_batch", (["order-1"],), {}),
    ],
)
def test_exchange_mode_rejects_order_calls(method_name: str, args: tuple, kwargs: dict) -> None:
    client = _StubOrderClient(api_mode="exchange")

    with pytest.raises(InvalidRequestError):
        getattr(client, method_name)(*args, **kwargs)


@pytest.mark.parametrize(
    ("method_name", "endpoint", "payload"),
    [
        ("place_order", "/orders", {"product_id": "BTC-USD"}),
        ("preview_order", "/order_preview", {"product_id": "BTC-USD"}),
        ("edit_order_preview", "/order_edit_preview", {"order_id": "order-1"}),
        ("edit_order", "/order_edit", {"order_id": "order-1"}),
        ("close_position", "/close_position", {"product_id": "BTC-USD"}),
    ],
)
def test_order_calls_dispatch_to_request(
    method_name: str, endpoint: str, payload: dict[str, Any]
) -> None:
    client = _StubOrderClient(responses=[{"ok": True}])

    result = getattr(client, method_name)(payload)

    assert result == {"ok": True}
    assert client.calls == [("POST", endpoint, payload)]


def test_list_orders_falls_back_to_open_on_not_found() -> None:
    client = _StubOrderClient(
        responses=[
            NotFoundError("missing"),
            {"orders": ["open"]},
        ]
    )

    result = client.list_orders(product_id="BTC-USD")

    assert result == {"orders": ["open"]}
    assert client.calls[0][1] == "/orders_historical?product_id=BTC-USD"
    assert client.calls[1][1] == "/orders?product_id=BTC-USD"


def test_list_orders_returns_empty_on_method_not_allowed() -> None:
    client = _StubOrderClient(
        responses=[
            NotFoundError("missing"),
            BrokerageError("Method not allowed"),
        ]
    )

    result = client.list_orders(product_id="BTC-USD")

    assert result == {"orders": []}
    assert client.calls[0][1] == "/orders_historical?product_id=BTC-USD"
    assert client.calls[1][1] == "/orders?product_id=BTC-USD"


def test_list_orders_exchange_mode_uses_open_endpoint() -> None:
    client = _StubOrderClient(api_mode="exchange", responses=[{"orders": ["open"]}])

    result = client.list_orders(product_id="BTC-USD")

    assert result == {"orders": ["open"]}
    assert client.calls == [("GET", "/orders?product_id=BTC-USD", None)]


def test_list_orders_batch_requires_ids() -> None:
    client = _StubOrderClient()

    with pytest.raises(InvalidRequestError):
        client.list_orders_batch([])


def test_list_orders_batch_builds_query() -> None:
    client = _StubOrderClient(responses=[{"orders": []}])

    result = client.list_orders_batch(["order-1", "order-2"], cursor="c1", limit=2)

    assert result == {"orders": []}
    path = client.calls[0][1]
    query = parse_qs(urlparse(path).query)
    assert query["order_ids"] == ["order-1", "order-2"]
    assert query["cursor"] == ["c1"]
    assert query["limit"] == ["2"]


def test_cancel_orders_dispatches_batch_cancel() -> None:
    client = _StubOrderClient(responses=[{"result": "ok"}])

    result = client.cancel_orders(["order-1", "order-2"])

    assert result == {"result": "ok"}
    assert client.calls == [("POST", "/orders_batch_cancel", {"order_ids": ["order-1", "order-2"]})]


def test_get_order_historical_calls_order_endpoint() -> None:
    client = _StubOrderClient(responses=[{"order_id": "order-1"}])

    result = client.get_order_historical("order-1")

    assert result == {"order_id": "order-1"}
    assert client.calls == [("GET", "/orders/order-1", None)]


def test_list_orders_historical_uses_query() -> None:
    client = _StubOrderClient(responses=[{"orders": []}])

    result = client.list_orders(product_id="BTC-USD", status="open")

    assert result == {"orders": []}
    path = client.calls[0][1]
    query = parse_qs(urlparse(path).query)
    assert query["product_id"] == ["BTC-USD"]
    assert query["status"] == ["open"]


def test_list_orders_fallback_raises_non_method_not_allowed() -> None:
    client = _StubOrderClient(
        responses=[
            NotFoundError("missing"),
            BrokerageError("bad gateway"),
        ]
    )

    with pytest.raises(BrokerageError, match="bad gateway"):
        client.list_orders(product_id="BTC-USD")


def test_list_orders_exchange_mode_includes_query() -> None:
    client = _StubOrderClient(api_mode="exchange", responses=[{"orders": []}])

    result = client.list_orders(product_id="BTC-USD", limit=2)

    assert result == {"orders": []}
    path = client.calls[0][1]
    query = parse_qs(urlparse(path).query)
    assert query["product_id"] == ["BTC-USD"]
    assert query["limit"] == ["2"]


def test_list_orders_batch_filters_ids_and_limit_zero() -> None:
    client = _StubOrderClient(responses=[{"orders": []}])

    result = client.list_orders_batch(["", None, "order-1", 123], limit=0)

    assert result == {"orders": []}
    path = client.calls[0][1]
    query = parse_qs(urlparse(path).query)
    assert query["order_ids"] == ["order-1", "123"]
    assert query["limit"] == ["0"]


def test_list_fills_builds_query() -> None:
    client = _StubOrderClient(responses=[{"fills": []}])

    result = client.list_fills(product_id="BTC-USD", limit=3)

    assert result == {"fills": []}
    path = client.calls[0][1]
    query = parse_qs(urlparse(path).query)
    assert query["product_id"] == ["BTC-USD"]
    assert query["limit"] == ["3"]

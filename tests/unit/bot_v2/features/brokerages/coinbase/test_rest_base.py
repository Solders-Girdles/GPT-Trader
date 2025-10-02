from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
import types

import pytest

from bot_v2.features.brokerages.coinbase.rest import base
from bot_v2.features.brokerages.core.interfaces import (
    InvalidRequestError,
    MarketType,
    OrderSide,
    OrderType,
    Product,
    TimeInForce,
)
from bot_v2.errors import ValidationError


class DummyClient:
    def __init__(self) -> None:
        self.place_payloads: list[dict[str, Any]] = []
        self.preview_payloads: list[dict[str, Any]] = []
        self.list_orders_payload: list[dict[str, Any]] = []
        self.products: list[dict[str, Any]] = []
        self.raise_invalid: bool = False
        self.raise_duplicate: bool = False

    def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.raise_invalid:
            raise InvalidRequestError("invalid")
        if self.raise_duplicate:
            raise InvalidRequestError("duplicate client_order_id")
        self.place_payloads.append(payload)
        return {"id": "order-1", **payload}

    def preview_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.preview_payloads.append(payload)
        return {"preview": payload}

    def list_orders(self, **params: Any) -> dict[str, Any]:
        self.list_orders_payload.append(params)
        return {
            "orders": [
                {
                    "client_order_id": params.get("client_order_id", "abc"),
                    "id": "existing",
                    "created_at": "2025-01-15T12:00:00Z",
                    "product_id": params.get("product_id", "BTC-USD"),
                    "type": "limit",
                    "side": "buy",
                    "size": "0.02",
                }
            ]
        }

    def get_products(self) -> dict[str, Any]:
        return {"products": self.products}


class DummyEventStore:
    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []
        self.positions: list[dict[str, Any]] = []

    def append_metric(self, *, bot_id: str, metrics: dict[str, Any]) -> None:
        self.metrics.append({"bot": bot_id, **metrics})

    def append_position(self, *, bot_id: str, position: dict[str, Any]) -> None:
        self.positions.append({"bot": bot_id, **position})


class DummyMarketData:
    def __init__(self) -> None:
        self.marks: dict[str, Decimal] = {}

    def get_mark(self, symbol: str) -> Decimal | None:
        return self.marks.get(symbol)

    def set_mark(self, symbol: str, price: Decimal) -> None:
        self.marks[symbol] = price


class DummyProductCatalog:
    def __init__(self, product: Product) -> None:
        self.product = product

    def get(self, client: DummyClient, symbol: str) -> Product:  # noqa: ARG002
        return self.product

    def get_funding(self, client: DummyClient, symbol: str) -> tuple[Decimal, datetime]:  # noqa: ARG002
        return Decimal("0.0001"), datetime.utcnow() + timedelta(hours=8)


class DummyFundingCalculator:
    def __init__(self, delta: Decimal) -> None:
        self.delta = delta

    def accrue_if_due(self, **kwargs: Any) -> Decimal:  # noqa: ARG002
        return self.delta


class DummyRestService(base.CoinbaseRestServiceBase):
    def __init__(self, product: Product) -> None:
        client = DummyClient()
        endpoints = types.SimpleNamespace()
        market = DummyMarketData()
        catalog = DummyProductCatalog(product)
        event_store = DummyEventStore()
        super().__init__(
            client=client,
            endpoints=endpoints,
            config=types.SimpleNamespace(),
            product_catalog=catalog,
            market_data=market,
            event_store=event_store,
        )
        self._quotes: dict[str, Any] = {}
        self._funding_calculator = DummyFundingCalculator(Decimal("0"))

    def get_product(self, symbol: str) -> Product:  # type: ignore[override]
        return self.product_catalog.product

    def get_rest_quote(self, product_id: str) -> Any:  # type: ignore[override]
        return self._quotes.get(product_id)


@pytest.fixture
def product() -> Product:
    return Product(
        symbol="BTC-USD",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.SPOT,
        min_size=Decimal("0.01"),
        step_size=Decimal("0.01"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.50"),
    )


@pytest.fixture
def service(product: Product) -> DummyRestService:
    svc = DummyRestService(product)
    svc._quotes["BTC-USD"] = types.SimpleNamespace(last=Decimal("20000"))
    return svc


def test_build_order_payload_limit(service: DummyRestService) -> None:
    payload = service._build_order_payload(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.015"),
        price=Decimal("20000.123"),
        stop_price=None,
        tif=TimeInForce.IOC,
        client_id="client-1",
        reduce_only=False,
        leverage=None,
        post_only=True,
    )
    config = payload["order_configuration"]["limit_limit_ioc"]
    assert config["limit_price"] == "20000.00"
    assert payload["post_only"] is True
    assert payload["time_in_force"] == "ioc"


def test_build_order_payload_market_raises_on_small_size(service: DummyRestService) -> None:
    with pytest.raises(ValidationError):
        service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.001"),
            price=None,
            stop_price=None,
            tif=TimeInForce.IOC,
            client_id=None,
            reduce_only=None,
            leverage=None,
            post_only=False,
        )


def test_build_order_payload_checks_notional(service: DummyRestService, product: Product) -> None:
    product.min_notional = Decimal("100000")
    with pytest.raises(ValidationError):
        service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.02"),
            price=None,
            stop_price=None,
            tif=TimeInForce.IOC,
            client_id=None,
            reduce_only=None,
            leverage=None,
            post_only=False,
        )


def test_execute_order_payload_duplicate_handles_existing(service: DummyRestService) -> None:
    service.client.raise_duplicate = True
    payload = {"product_id": "BTC-USD", "quantity": "0.02", "client_order_id": "abc"}
    result = service._execute_order_payload("BTC-USD", payload, client_id="abc")
    assert result.id == "existing"


def test_execute_order_payload_success(service: DummyRestService) -> None:
    payload = {"product_id": "BTC-USD", "quantity": "0.02"}
    result = service._execute_order_payload("BTC-USD", payload, client_id=None)
    assert result.id == "order-1"


def test_execute_order_payload_invalid_request_raises(service: DummyRestService) -> None:
    service.client.raise_invalid = True
    payload = {"product_id": "BTC-USD", "quantity": "0.01", "client_order_id": "dup"}
    with pytest.raises(InvalidRequestError):
        service._execute_order_payload("BTC-USD", payload, client_id="dup")


def test_execute_order_payload_general_exception(service: DummyRestService) -> None:
    def boom(payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
        raise RuntimeError("boom")

    service.client.place_order = boom  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        service._execute_order_payload("BTC-USD", {"product_id": "BTC-USD", "quantity": "0.1"}, client_id=None)


def test_update_position_metrics_records_events(service: DummyRestService) -> None:
    service._positions["BTC-USD"] = types.SimpleNamespace(
        quantity=Decimal("1"),
        entry_price=Decimal("20000"),
        side="long",
        realized_pnl=Decimal("0"),
        get_unrealized_pnl=lambda mark: mark - Decimal("20000"),
    )
    service.market_data.set_mark("BTC-USD", Decimal("21000"))
    service._funding_calculator = DummyFundingCalculator(Decimal("15"))
    service.product_catalog.product.next_funding_time = datetime.utcnow() + timedelta(hours=8)

    service._update_position_metrics("BTC-USD")

    assert service._positions["BTC-USD"].realized_pnl == Decimal("15")
    assert service._event_store.metrics
    assert service._event_store.positions

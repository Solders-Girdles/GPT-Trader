"""Coinbase trading and order management tests."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.core.interfaces import (
    InvalidRequestError as CoreInvalidRequestError,
    MarketType,
    Order,
    OrderSide,
    OrderType,
    Position,
    Product,
    TimeInForce,
)

from tests.unit.bot_v2.features.brokerages.coinbase.minimal_brokerage import (
    MinimalCoinbaseBrokerage,
)
from tests.unit.bot_v2.features.brokerages.coinbase.test_helpers import (
    CLIENT_ENDPOINT_CASES,
    _decode_body,
    make_adapter,
    make_client,
)


pytestmark = pytest.mark.endpoints


class TestCoinbaseTrading:
    @pytest.mark.parametrize("case", CLIENT_ENDPOINT_CASES, ids=lambda c: c["id"])
    def test_client_trading_endpoints(self, case: dict[str, Any]) -> None:
        client = make_client(case.get("api_mode", "advanced"))
        recorded: dict[str, Any] = {}

        def transport(method, url, headers, body, timeout):
            recorded["method"] = method
            recorded["url"] = url
            recorded["body"] = body
            return 200, {}, json.dumps(case.get("response", {}))

        client.set_transport_for_testing(transport)

        result = getattr(client, case["method"])(*case.get("args", ()), **case.get("kwargs", {}))

        assert recorded["method"] == case["expected_method"]
        parsed = urlparse(recorded["url"])
        assert parsed.path.endswith(case["expected_path"])

        expected_query = case.get("expected_query")
        if expected_query is not None:
            assert parse_qs(parsed.query) == expected_query
        else:
            assert parsed.query in ("", None)

        expected_payload = case.get("expected_payload")
        if expected_payload is not None:
            assert _decode_body(recorded.get("body")) == expected_payload
        else:
            assert not recorded.get("body")

        expected_result = case.get("expected_result")
        if expected_result is not None:
            assert result == expected_result

    def test_list_orders_pagination(self) -> None:
        client = make_client()
        call_count = 0

        def transport(method, url, headers, body, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (
                    200,
                    {},
                    json.dumps({"orders": [{"id": "ord1"}, {"id": "ord2"}], "cursor": "next"}),
                )
            return 200, {}, json.dumps({"orders": [{"id": "ord3"}], "cursor": None})

        client.set_transport_for_testing(transport)

        collected = list(
            client.paginate(
                path="/api/v3/brokerage/orders/historical",
                params={},
                items_key="orders",
            )
        )

        assert [item["id"] for item in collected] == ["ord1", "ord2", "ord3"]
        assert call_count == 2

    def test_list_orders_batch_requires_order_ids(self) -> None:
        client = make_client()
        with pytest.raises(CoreInvalidRequestError):
            client.list_orders_batch([])

    def test_adapter_list_orders_batch_normalises_orders(self, monkeypatch) -> None:
        adapter = make_adapter()
        captured: dict[str, Any] = {}

        def fake_batch(order_ids, cursor=None, limit=None):
            captured["order_ids"] = order_ids
            captured["cursor"] = cursor
            captured["limit"] = limit
            return {
                "orders": [
                    {
                        "order_id": "ord-1",
                        "product_id": "BTC-USD",
                        "side": "BUY",
                        "type": "limit",
                        "size": "1",
                        "time_in_force": "GTC",
                        "status": "open",
                        "created_at": "2024-01-01T00:00:00",
                    }
                ]
            }

        monkeypatch.setattr(adapter.client, "list_orders_batch", fake_batch)

        orders = adapter.list_orders_batch(["ord-1"], cursor="cursor-1", limit=25)

        assert captured["order_ids"] == ["ord-1"]
        assert captured["cursor"] == "cursor-1"
        assert captured["limit"] == 25
        assert len(orders) == 1
        assert isinstance(orders[0], Order)
        assert orders[0].id == "ord-1"

    def test_transaction_summary_blocked_in_exchange_mode(self) -> None:
        client = make_client("exchange")
        with pytest.raises(CoreInvalidRequestError):
            client.get_transaction_summary()

    def test_make_broker_configures_coinbase_client(self) -> None:
        broker = make_adapter()
        assert broker.client.base_url.startswith("https://api")

    def _adapter_with_product(self) -> CoinbaseBrokerage:
        config = APIConfig(
            api_key="k",
            api_secret="s",
            passphrase=None,
            base_url="https://api.coinbase.com",
            api_mode="advanced",
            sandbox=False,
            enable_derivatives=True,
            auth_type="HMAC",
        )
        adapter = CoinbaseBrokerage(config)

        class Catalog:
            def get(self, client, symbol):
                return Product(
                    symbol=symbol,
                    base_asset="BTC",
                    quote_asset="USD",
                    market_type=MarketType.PERPETUAL,
                    min_size=Decimal("0.001"),
                    step_size=Decimal("0.001"),
                    min_notional=None,
                    price_increment=Decimal("0.01"),
                )

            def get_funding(self, client, symbol):  # pragma: no cover - simple stub
                return None, None

        adapter.product_catalog = Catalog()
        return adapter

    def test_close_position_uses_fallback_when_requested(self, monkeypatch) -> None:
        adapter = self._adapter_with_product()
        positions = [
            Position(
                symbol="BTC-USD",
                quantity=Decimal("0.75"),
                entry_price=Decimal("50000"),
                mark_price=Decimal("50500"),
                unrealized_pnl=Decimal("375"),
                realized_pnl=Decimal("0"),
                leverage=3,
                side="long",
            )
        ]
        monkeypatch.setattr(adapter, "list_positions", lambda: positions)

        place_calls: list[dict[str, Any]] = []

        def fake_place_order(**kwargs):
            place_calls.append(kwargs)
            return {"order": kwargs}

        monkeypatch.setattr(adapter, "place_order", fake_place_order)

        def fake_close(symbol, quantity, reduce_only, positions_override, fallback):
            assert symbol == "BTC-USD"
            assert reduce_only is True
            assert positions_override == positions
            return fallback(OrderSide.SELL, Decimal("0.75"), reduce_only)

        monkeypatch.setattr(adapter.rest_service, "close_position", fake_close)

        result = adapter.close_position("BTC-USD", reduce_only=True)
        assert result == {"order": place_calls[0]}

        assert place_calls
        call = place_calls[0]
        assert call["symbol"] == "BTC-USD"
        assert call["side"] is OrderSide.SELL
        assert call["order_type"] is OrderType.MARKET
        assert call["quantity"] == Decimal("0.75")
        assert call["reduce_only"] is True

    def make_minimal_brokerage(self) -> MinimalCoinbaseBrokerage:
        config = APIConfig(
            api_key="test",
            api_secret="test",
            passphrase="test",
            base_url="https://api.sandbox.coinbase.com",
            sandbox=True,
            enable_derivatives=True,
            api_mode="advanced",
            auth_type="HMAC",
        )
        return MinimalCoinbaseBrokerage(config)

    def test_time_in_force_default_gtc(self) -> None:
        adapter = self.make_minimal_brokerage()
        order = adapter.place_order(
            symbol="BTC-PERP",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.01"),
            limit_price=Decimal("50000"),
        )
        assert order.tif is TimeInForce.GTC

    def test_time_in_force_ioc_mapping(self) -> None:
        adapter = self.make_minimal_brokerage()
        order = adapter.place_order(
            symbol="BTC-PERP",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.01"),
            limit_price=Decimal("50000"),
            tif="IOC",
        )
        assert order is not None

    def test_limit_order_accepts_post_only_flag(self) -> None:
        adapter = self.make_minimal_brokerage()
        order = adapter.place_order(
            symbol="BTC-PERP",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.01"),
            limit_price=Decimal("50000"),
            post_only=True,
        )
        assert order is not None

    def test_market_order_does_not_require_post_only(self) -> None:
        adapter = self.make_minimal_brokerage()
        order = adapter.place_order(
            symbol="BTC-PERP",
            side="sell",
            order_type="market",
            quantity=Decimal("0.02"),
        )
        assert order is not None

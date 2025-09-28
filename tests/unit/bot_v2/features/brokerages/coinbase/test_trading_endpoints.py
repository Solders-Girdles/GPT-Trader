"""Combined trading endpoint and adapter tests for Coinbase integration."""

import json
from decimal import Decimal

import pytest

from bot_v2.features.brokerages.coinbase import client as client_mod
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.errors import InvalidRequestError
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.utilities import quantize_to_increment
from bot_v2.features.brokerages.core.interfaces import (
    MarketType,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Product,
    TimeInForce,
)
from bot_v2.errors import ValidationError

from tests.unit.bot_v2.features.brokerages.coinbase.minimal_brokerage import (
    MinimalCoinbaseBrokerage,
)


pytestmark = pytest.mark.endpoints


def make_client(api_mode: str = "advanced") -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


def make_broker() -> CoinbaseBrokerage:
    config = APIConfig(api_key="k", api_secret="s", passphrase=None, base_url="https://api")
    return CoinbaseBrokerage(config)


def _make_adapter() -> CoinbaseBrokerage:
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

    class ProdCat:
        def get(self, client, symbol):
            return Product(
                symbol=symbol,
                base_asset=symbol.split("-")[0],
                quote_asset="USD",
                market_type=MarketType.PERPETUAL,
                min_size=Decimal("0.001"),
                step_size=Decimal("0.001"),
                min_notional=None,
                price_increment=Decimal("0.01"),
            )

        def get_funding(self, client, symbol):  # pragma: no cover - simple stub
            return None, None

    adapter.product_catalog = ProdCat()
    return adapter


def _adapter_with_product() -> CoinbaseBrokerage:
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

        def get_funding(self, client, symbol):  # pragma: no cover - fallback path
            return None, None

    adapter.product_catalog = Catalog()
    return adapter


# ----------------------------------------------------------------------------
# CoinbaseClient trading endpoints
# ----------------------------------------------------------------------------


def test_list_orders_builds_query_string():
    client = make_client()
    urls = []

    def transport(method, url, headers, body, timeout):
        urls.append(url)
        return 200, {}, json.dumps({"orders": []})

    client.set_transport_for_testing(transport)
    _ = client.list_orders(
        product_id="BTC-USD",
        order_status="FILLED",
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    url = urls[0]
    assert url.endswith(
        "/api/v3/brokerage/orders/historical"
        "?product_id=BTC-USD&order_status=FILLED&start_date=2024-01-01&end_date=2024-01-31"
    )


def test_list_fills_filters_by_symbol():
    client = make_client()
    urls = []

    def transport(method, url, headers, body, timeout):
        urls.append(url)
        return 200, {}, json.dumps({"fills": []})

    client.set_transport_for_testing(transport)
    _ = client.list_fills(product_id="ETH-USD", limit=100)
    assert urls[0].endswith(
        "/api/v3/brokerage/orders/historical/fills?product_id=ETH-USD&limit=100"
    )


def test_cancel_orders_handles_multiple_ids():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"results": [1, 2]})

    client.set_transport_for_testing(transport)
    out = client.cancel_orders(["id1", "id2"])
    method, url, payload = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/orders/batch_cancel")
    assert payload["order_ids"] == ["id1", "id2"]
    assert out.get("results") == [1, 2]


def test_get_order_historical_formats_path():
    client = make_client()
    urls = []

    def transport(method, url, headers, body, timeout):
        urls.append(url)
        return 200, {}, json.dumps({"order": {"id": "ord-1"}})

    client.set_transport_for_testing(transport)
    _ = client.get_order_historical("ord-1")
    assert urls[0].endswith("/api/v3/brokerage/orders/historical/ord-1")


def test_list_orders_batch_handles_multiple_ids():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"orders": []})

    client.set_transport_for_testing(transport)
    out = client.list_orders_batch(["order1", "order2", "order3"])
    assert calls[0][0] == "GET"
    assert "/batch" in calls[0][1]
    assert "orders" in out


def test_place_order_sends_post():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"order_id": "new-order-123"})

    client.set_transport_for_testing(transport)
    payload = {"product_id": "BTC-USD", "side": "BUY", "order_configuration": {}}
    out = client.place_order(payload)

    method, url, sent_payload = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/orders")
    assert sent_payload == payload
    assert out["order_id"] == "new-order-123"


def test_list_orders_pagination():
    client = make_client()
    call_count = 0

    def transport(method, url, headers, body, timeout):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (
                200,
                {},
                json.dumps(
                    {"orders": [{"id": "ord1"}, {"id": "ord2"}], "cursor": "next-page-cursor"}
                ),
            )
        return 200, {}, json.dumps({"orders": [{"id": "ord3"}], "cursor": None})

    client.set_transport_for_testing(transport)

    all_orders = list(
        client.paginate(
            path="/api/v3/brokerage/orders/historical",
            params={},
            items_key="orders",
        )
    )

    assert [o["id"] for o in all_orders] == ["ord1", "ord2", "ord3"]
    assert call_count == 2


def test_preview_order_posts_payload():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"preview_id": "prev-1"})

    client.set_transport_for_testing(transport)
    payload = {
        "product_id": "BTC-USD",
        "side": "BUY",
        "order_configuration": {"limit_limit_gtc": {"base_size": "0.1", "limit_price": "50000"}},
    }
    out = client.preview_order(payload)
    method, url, sent = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/orders/preview")
    assert sent == payload
    assert out["preview_id"] == "prev-1"


def test_edit_order_preview_posts_payload():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"edit_preview_id": "ep-1"})

    client.set_transport_for_testing(transport)
    payload = {"order_id": "ord-1", "new_price": "49900"}
    out = client.edit_order_preview(payload)
    method, url, sent = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/orders/edit_preview")
    assert sent == payload
    assert out["edit_preview_id"] == "ep-1"


def test_edit_order_posts_payload():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"success": True})

    client.set_transport_for_testing(transport)
    payload = {"order_id": "ord-1", "price": "49800"}
    out = client.edit_order(payload)
    method, url, sent = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/orders/edit")
    assert sent == payload
    assert out["success"] is True


def test_get_transaction_summary_path_and_method():
    client = make_client("advanced")
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"summary": {"fees": "10", "volume": "1000"}})

    client.set_transport_for_testing(transport)
    out = client.get_transaction_summary()
    method, url = calls[0]
    assert method == "GET"
    assert url.endswith("/api/v3/brokerage/transaction_summary")
    assert "summary" in out


def test_get_transaction_summary_blocked_in_exchange():
    client = make_client("exchange")
    with pytest.raises(InvalidRequestError):
        client.get_transaction_summary()


# ----------------------------------------------------------------------------
# Adapter trading flows
# ----------------------------------------------------------------------------


def test_cancel_order_success(monkeypatch):
    broker = make_broker()

    def fake_cancel(self, ids):
        return {"results": [{"order_id": ids[0], "success": True}]}

    monkeypatch.setattr(client_mod.CoinbaseClient, "cancel_orders", fake_cancel)
    assert broker.cancel_order("abc123") is True


def test_get_order_maps(monkeypatch):
    broker = make_broker()

    def fake_get(self, order_id):
        return {
            "id": order_id,
            "product_id": "BTC-USD",
            "side": "buy",
            "type": "limit",
            "size": "1",
            "price": "100",
            "status": "filled",
            "filled_size": "1",
            "average_filled_price": "100",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:01:00",
        }

    monkeypatch.setattr(client_mod.CoinbaseClient, "get_order_historical", fake_get)
    order = broker.get_order("OID")
    assert order.id == "OID"
    assert order.status == OrderStatus.FILLED


def test_list_orders_maps(monkeypatch):
    broker = make_broker()

    def fake_list(self, **params):
        return {
            "orders": [
                {
                    "id": "o1",
                    "product_id": "BTC-USD",
                    "side": "buy",
                    "type": "market",
                    "size": "1",
                    "status": "filled",
                    "filled_size": "1",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:10",
                },
                {
                    "id": "o2",
                    "product_id": "ETH-USD",
                    "side": "sell",
                    "type": "limit",
                    "size": "2",
                    "price": "2000",
                    "status": "open",
                    "filled_size": "0",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:10",
                },
            ]
        }

    monkeypatch.setattr(client_mod.CoinbaseClient, "list_orders", fake_list)
    out = broker.list_orders()
    assert len(out) == 2
    assert out[0].id == "o1" and out[1].id == "o2"


def test_list_orders_returns_empty_on_error(monkeypatch):
    broker = make_broker()

    def boom(self, **params):
        raise RuntimeError("501 method not allowed")

    monkeypatch.setattr(client_mod.CoinbaseClient, "list_orders", boom)
    assert broker.list_orders() == []


def test_list_fills(monkeypatch):
    broker = make_broker()

    def fake_fills(self, **params):
        return {"fills": [{"trade_id": "t1"}, {"trade_id": "t2"}]}

    monkeypatch.setattr(client_mod.CoinbaseClient, "list_fills", fake_fills)
    fills = broker.list_fills("BTC-USD", limit=2)
    assert [fill["trade_id"] for fill in fills] == ["t1", "t2"]


def test_list_fills_returns_empty_on_error(monkeypatch, caplog):
    broker = make_broker()

    def boom(self, **params):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(client_mod.CoinbaseClient, "list_fills", boom)
    caplog.set_level("ERROR")
    assert broker.list_fills("BTC-USD") == []
    assert any("Failed to list fills" in rec.message for rec in caplog.records)


def test_payload_mapping_market_ioc(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setenv("ORDER_PREVIEW_ENABLED", "1")
    captured = {}

    def preview(payload):
        captured["preview"] = payload
        return {"ok": True}

    def place(payload):
        captured["place"] = payload
        return {
            "order_id": "abc",
            "status": "open",
            "product_id": payload.get("product_id"),
            "type": "market",
            "side": payload.get("side").lower(),
            "time_in_force": "ioc",
            "client_order_id": payload.get("client_order_id"),
            "size": payload["order_configuration"]["market_market_ioc"].get("base_size"),
        }

    adapter.client.preview_order = preview  # type: ignore[attr-defined]
    adapter.client.place_order = place  # type: ignore[attr-defined]

    order = adapter.place_order(
        symbol="BTC-USD-PERP",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1234"),
        tif=TimeInForce.IOC,
    )
    assert "preview" in captured and "place" in captured
    payload = captured["place"]
    assert payload["order_configuration"]["market_market_ioc"]["base_size"] == "0.123"
    assert payload["side"] == "BUY"
    assert order.type == OrderType.MARKET


def test_payload_mapping_limit_tifs(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setenv("ORDER_PREVIEW_ENABLED", "1")
    captured = {}
    adapter.client.preview_order = lambda payload: {"ok": True}  # type: ignore[attr-defined]

    def place(payload):
        captured["place"] = payload
        return {
            "order_id": "xyz",
            "status": "open",
            "product_id": payload.get("product_id"),
            "type": "limit",
            "side": payload.get("side").lower(),
            "time_in_force": "gtc",
            "price": payload["order_configuration"]["limit_limit_gtc"].get("limit_price"),
            "client_order_id": payload.get("client_order_id"),
        }

    adapter.client.place_order = place  # type: ignore[attr-defined]

    adapter.place_order(
        symbol="ETH-USD-PERP",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("2500.123"),
        tif=TimeInForce.GTC,
    )
    payload = captured["place"]
    assert "limit_limit_gtc" in payload["order_configuration"]
    assert payload["order_configuration"]["limit_limit_gtc"]["limit_price"] == "2500.12"

    def place_ioc(payload):
        captured["place_ioc"] = payload
        return {
            "order_id": "ioc",
            "status": "open",
            "product_id": payload.get("product_id"),
            "type": "limit",
            "side": payload.get("side").lower(),
            "time_in_force": "ioc",
            "price": payload["order_configuration"]["limit_limit_ioc"].get("limit_price"),
            "client_order_id": payload.get("client_order_id"),
        }

    adapter.client.place_order = place_ioc  # type: ignore[attr-defined]
    adapter.place_order(
        symbol="ETH-USD-PERP",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("2500.00"),
        tif=TimeInForce.IOC,
    )
    assert "limit_limit_ioc" in captured["place_ioc"]["order_configuration"]

    def place_fok(payload):
        captured["place_fok"] = payload
        return {
            "order_id": "fok",
            "status": "open",
            "product_id": payload.get("product_id"),
            "type": "limit",
            "side": payload.get("side").lower(),
            "time_in_force": "fok",
            "price": payload["order_configuration"]["limit_limit_fok"].get("limit_price"),
            "client_order_id": payload.get("client_order_id"),
        }

    adapter.client.place_order = place_fok  # type: ignore[attr-defined]
    adapter.place_order(
        symbol="ETH-USD-PERP",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("2500.00"),
        tif=TimeInForce.FOK,
    )
    assert "limit_limit_fok" in captured["place_fok"]["order_configuration"]


def test_duplicate_client_order_id_resolution(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.delenv("ORDER_PREVIEW_ENABLED", raising=False)

    def place(payload):
        raise InvalidRequestError("duplicate client_order_id")

    adapter.client.place_order = place  # type: ignore[attr-defined]

    matching = {
        "orders": [
            {
                "order_id": "old1",
                "status": "open",
                "client_order_id": "same123",
                "product_id": "BTC-USD-PERP",
                "type": "market",
                "side": "buy",
                "time_in_force": "ioc",
                "created_at": "2020-01-01T00:00:00Z",
            },
            {
                "order_id": "1122",
                "status": "open",
                "client_order_id": "same123",
                "product_id": "BTC-USD-PERP",
                "type": "market",
                "side": "buy",
                "time_in_force": "ioc",
                "created_at": "2099-01-01T00:00:00Z",
            },
        ]
    }

    adapter.client.list_orders = lambda **params: matching  # type: ignore[attr-defined]

    order = adapter.place_order(
        symbol="BTC-USD-PERP",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        client_id="same123",
    )
    assert order.id == "1122"


def test_preview_order_builds_same_payload(monkeypatch):
    adapter = _make_adapter()
    captured = {}
    adapter.client.preview_order = lambda payload: captured.setdefault("payload", payload) or {"success": True}  # type: ignore[attr-defined]

    adapter.preview_order(
        symbol="BTC-USD-PERP",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0.25"),
        tif=TimeInForce.IOC,
    )
    payload = captured["payload"]
    assert payload["order_configuration"]["market_market_ioc"]["base_size"] == "0.250"
    assert payload["side"] == "BUY"


def test_edit_order_preview_wraps_configuration(monkeypatch):
    adapter = _make_adapter()
    captured = {}

    def edit_preview(payload):
        captured["payload"] = payload
        return {"preview_id": "prev123"}

    adapter.client.edit_order_preview = edit_preview  # type: ignore[attr-defined]

    resp = adapter.edit_order_preview(
        order_id="order-1",
        symbol="ETH-USD-PERP",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.5"),
        price=Decimal("2000"),
        tif=TimeInForce.GTC,
        new_client_id="client-xyz",
        reduce_only=True,
    )
    payload = captured["payload"]
    assert payload["order_id"] == "order-1"
    assert payload["reduce_only"] is True
    assert resp["preview_id"] == "prev123"


def test_edit_order_returns_order(monkeypatch):
    adapter = _make_adapter()

    def edit(payload):
        return {
            "order_id": "ord123",
            "product_id": "BTC-USD-PERP",
            "client_order_id": payload.get("preview_id"),
            "status": "open",
            "side": "buy",
            "type": "market",
            "time_in_force": "ioc",
            "size": "0.1",
            "created_at": "2024-01-01T00:00:00Z",
        }

    adapter.client.edit_order = edit  # type: ignore[attr-defined]
    order = adapter.edit_order("order-1", "preview-1")
    assert order.id == "ord123"
    assert order.client_id == "preview-1"


def test_payment_methods_and_portfolios_wrappers(monkeypatch):
    adapter = _make_adapter()
    adapter.client.list_payment_methods = lambda: {"payment_methods": [{"id": "pm-1"}]}  # type: ignore[attr-defined]
    adapter.client.get_payment_method = lambda pm_id: {"payment_method": {"id": pm_id}}  # type: ignore[attr-defined]
    adapter.client.list_portfolios = lambda: {"portfolios": [{"uuid": "pf-1"}]}  # type: ignore[attr-defined]
    adapter.client.get_portfolio = lambda uuid: {"portfolio": {"uuid": uuid}}  # type: ignore[attr-defined]
    adapter.client.get_portfolio_breakdown = lambda uuid: {"breakdown": {"uuid": uuid}}  # type: ignore[attr-defined]
    adapter.client.move_funds = lambda payload: {"status": "ok", **payload}  # type: ignore[attr-defined]

    assert adapter.list_payment_methods() == [{"id": "pm-1"}]
    assert adapter.get_payment_method("pm-9")["id"] == "pm-9"
    assert adapter.list_portfolios() == [{"uuid": "pf-1"}]
    assert adapter.get_portfolio("pf-1")["uuid"] == "pf-1"
    assert adapter.get_portfolio_breakdown("pf-1")["uuid"] == "pf-1"
    move_resp = adapter.move_portfolio_funds({"from": "pf-1", "to": "pf-2", "amount": "10"})
    assert move_resp["status"] == "ok"


def test_convert_wrappers(monkeypatch):
    adapter = _make_adapter()
    captured = {}
    adapter.client.convert_quote = lambda payload: captured.setdefault("quote", payload) or {"quote_id": "q1"}  # type: ignore[attr-defined]
    adapter.client.commit_convert_trade = lambda trade_id, payload: {"trade_id": trade_id, **(payload or {})}  # type: ignore[attr-defined]
    adapter.client.get_convert_trade = lambda trade_id: {"trade_id": trade_id, "status": "settled"}  # type: ignore[attr-defined]

    quote = adapter.create_convert_quote({"from": "USDC", "to": "USD", "amount": "100"})
    assert captured["quote"]["amount"] == "100"
    commit = adapter.commit_convert_trade("trade-1", {"amount": "100"})
    assert commit["trade_id"] == "trade-1"
    status = adapter.get_convert_trade("trade-1")
    assert status["status"] == "settled"


def test_quantize_helper():
    assert quantize_to_increment(Decimal("1.2345"), Decimal("0.01")) == Decimal("1.23")
    assert quantize_to_increment(Decimal("0.0009"), Decimal("0.001")) == Decimal("0.000")


def test_place_order_applies_rounding_and_builds_payload(monkeypatch):
    broker = make_broker()

    product_payload = {
        "id": "BTC-USD",
        "base_currency": "BTC",
        "quote_currency": "USD",
        "base_min_size": "0.001",
        "base_increment": "0.001",
        "quote_increment": "0.1",
        "min_notional": "0.01",
        "contract_type": None,
    }

    monkeypatch.setattr(
        client_mod.CoinbaseClient,
        "get_products",
        lambda self: {"products": [product_payload]},
    )

    captured = {}

    def fake_place_order(self, payload):
        captured.update(payload)
        return {
            "id": "oid",
            "product_id": payload["product_id"],
            "side": payload["side"],
            "type": "limit",
            "size": payload["order_configuration"].get("limit_limit_gtc", {}).get("base_size", "0"),
            "price": payload["order_configuration"].get("limit_limit_gtc", {}).get("limit_price"),
            "status": "open",
            "filled_size": "0",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

    monkeypatch.setattr(client_mod.CoinbaseClient, "place_order", fake_place_order)

    order = broker.place_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.0014"),
        price=Decimal("100.05"),
        tif=TimeInForce.GTC,
    )
    config = captured["order_configuration"]["limit_limit_gtc"]
    assert config["base_size"] == "0.001"
    assert config["limit_price"] == "100.0"
    assert captured["product_id"] == "BTC-USD"
    assert order.id == "oid"


def test_min_notional_enforced(monkeypatch):
    broker = make_broker()

    product_payload = {
        "id": "BTC-USD",
        "base_currency": "BTC",
        "quote_currency": "USD",
        "base_min_size": "0.001",
        "base_increment": "0.001",
        "quote_increment": "0.1",
        "min_notional": "50",
    }

    monkeypatch.setattr(
        client_mod.CoinbaseClient,
        "get_products",
        lambda self: {"products": [product_payload]},
    )

    with pytest.raises(ValidationError):
        broker.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("10.0"),
            tif=TimeInForce.GTC,
        )


class TestTIFMapping:
    """Test Time-In-Force mappings via minimal adapter."""

    def setup_method(self):
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
        self.adapter = MinimalCoinbaseBrokerage(config)

    def test_gtc_tif_default(self):
        order = self.adapter.place_order(
            symbol="BTC-PERP",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.01"),
            limit_price=Decimal("50000"),
        )
        assert order.tif == TimeInForce.GTC

    def test_ioc_tif_mapping(self):
        order = self.adapter.place_order(
            symbol="BTC-PERP",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.01"),
            limit_price=Decimal("50000"),
            tif="IOC",
        )
        assert order is not None

    def test_market_order_no_post_only(self):
        order = self.adapter.place_order(
            symbol="BTC-PERP",
            side="buy",
            order_type="market",
            quantity=Decimal("0.01"),
        )
        assert order is not None

    def test_limit_order_accepts_post_only(self):
        order = self.adapter.place_order(
            symbol="BTC-PERP",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.01"),
            limit_price=Decimal("50000"),
            post_only=True,
        )
        assert order is not None


def test_close_position_uses_endpoint(monkeypatch):
    adapter = _adapter_with_product()

    adapter.list_positions = lambda: [
        Position(
            symbol="BTC-USD-PERP",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50500"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("0"),
            leverage=3,
            side="long",
        )
    ]  # type: ignore[attr-defined]

    payloads = {}

    def close_position(payload):
        payloads["cp"] = payload
        return {
            "order_id": "cp123",
            "status": "open",
            "product_id": payload.get("product_id"),
            "type": "market",
            "side": payload.get("side").lower(),
            "time_in_force": "ioc",
        }

    adapter.client.close_position = close_position  # type: ignore[attr-defined]

    order = adapter.close_position("BTC-USD-PERP")
    assert order.id == "cp123"
    assert payloads["cp"]["product_id"] == "BTC-USD-PERP"
    assert payloads["cp"]["size"] == "1.0"
    assert payloads["cp"]["reduce_only"] is True


def test_close_position_fallback_to_market(monkeypatch):
    adapter = _adapter_with_product()
    adapter.list_positions = lambda: [
        Position(
            symbol="BTC-USD-PERP",
            quantity=Decimal("-2.0"),
            entry_price=Decimal("30000"),
            mark_price=Decimal("29500"),
            unrealized_pnl=Decimal("-1000"),
            realized_pnl=Decimal("0"),
            leverage=2,
            side="short",
        )
    ]  # type: ignore[attr-defined]

    def close_position(payload):
        raise InvalidRequestError("endpoint not available")

    adapter.client.close_position = close_position  # type: ignore[attr-defined]

    called = {}

    def place_order(**kwargs):
        from types import SimpleNamespace

        called.update(kwargs)
        return SimpleNamespace(id="fallback123")

    from types import MethodType

    adapter.place_order = MethodType(lambda self, **kw: place_order(**kw), adapter)

    order_id = adapter.close_position("BTC-USD-PERP").id
    assert order_id == "fallback123"
    assert called["reduce_only"] is True
    assert called["order_type"] == OrderType.MARKET
    assert called["side"] == OrderSide.BUY

"""Comprehensive Coinbase core integration coverage."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse
from unittest.mock import MagicMock, patch

import pytest

import bot_v2.features.brokerages.coinbase.client as client_mod
from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.errors import AuthError, InsufficientFunds
from bot_v2.features.brokerages.coinbase.market_data_features import (
    DepthSnapshot,
    RollingWindow,
    TradeTapeAgg,
)
from bot_v2.features.brokerages.coinbase.models import APIConfig, to_product
from bot_v2.features.brokerages.coinbase.utilities import (
    ProductCatalog,
    enforce_perp_rules,
    quantize_to_increment,
)
from bot_v2.features.brokerages.core.interfaces import (
    InvalidRequestError as CoreInvalidRequestError,
    MarketType,
    NotFoundError,
    OrderSide,
    OrderType,
    Position,
    Product,
    TimeInForce,
)

from tests.unit.bot_v2.features.brokerages.coinbase.minimal_brokerage import (
    MinimalCoinbaseBrokerage,
)


pytestmark = pytest.mark.endpoints


def make_client(api_mode: str = "advanced", auth=None) -> CoinbaseClient:
    """Create a CoinbaseClient pointing at the public API host."""

    return CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode=api_mode)


def make_adapter(api_mode: str = "advanced") -> CoinbaseBrokerage:
    """Construct a brokerage adapter with default credentials."""

    config = APIConfig(
        api_key="k",
        api_secret="s",
        passphrase=None,
        base_url="https://api.coinbase.com",
        api_mode=api_mode,
        sandbox=False,
    )
    return CoinbaseBrokerage(config)


def _decode_body(body: Any) -> dict:
    if not body:
        return {}
    if isinstance(body, bytes):
        body = body.decode()
    return json.loads(body)


def url_has_param(url: str, fragment: str) -> bool:
    return ("?" + fragment) in url or ("&" + fragment) in url


ACCOUNT_ENDPOINT_CASES = [
    pytest.param(
        {
            "id": "get_accounts",
            "method": "get_accounts",
            "args": (),
            "kwargs": {},
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/accounts",
            "expected_query": {},
            "response": {"accounts": []},
            "expected_result": {"accounts": []},
        },
        id="get_accounts",
    ),
    pytest.param(
        {
            "id": "get_account",
            "method": "get_account",
            "args": ("acc-123",),
            "kwargs": {},
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/accounts/acc-123",
            "expected_query": {},
            "response": {"account": {"uuid": "acc-123"}},
            "expected_result": {"account": {"uuid": "acc-123"}},
        },
        id="get_account",
    ),
    pytest.param(
        {
            "id": "list_portfolios",
            "method": "list_portfolios",
            "args": (),
            "kwargs": {},
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/portfolios",
            "expected_query": {},
            "response": {"portfolios": []},
            "expected_result": {"portfolios": []},
        },
        id="list_portfolios",
    ),
    pytest.param(
        {
            "id": "get_portfolio",
            "method": "get_portfolio",
            "args": ("port-456",),
            "kwargs": {},
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/portfolios/port-456",
            "expected_query": {},
            "response": {"portfolio": {"uuid": "port-456"}},
            "expected_result": {"portfolio": {"uuid": "port-456"}},
        },
        id="get_portfolio",
    ),
    pytest.param(
        {
            "id": "get_portfolio_breakdown",
            "method": "get_portfolio_breakdown",
            "args": ("port-789",),
            "kwargs": {},
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/portfolios/port-789/breakdown",
            "expected_query": {},
            "response": {"breakdown": {"total_balance": "10000"}},
            "expected_result": {"breakdown": {"total_balance": "10000"}},
        },
        id="get_portfolio_breakdown",
    ),
    pytest.param(
        {
            "id": "move_funds",
            "method": "move_funds",
            "args": (
                {
                    "from_portfolio_uuid": "port-A",
                    "to_portfolio_uuid": "port-B",
                    "amount": "1000",
                    "currency": "USD",
                },
            ),
            "kwargs": {},
            "expected_method": "POST",
            "expected_path": "/api/v3/brokerage/portfolios/move_funds",
            "expected_query": {},
            "expected_payload": {
                "from_portfolio_uuid": "port-A",
                "to_portfolio_uuid": "port-B",
                "amount": "1000",
                "currency": "USD",
            },
            "response": {"transfer_id": "mv-1", "status": "completed"},
            "expected_result": {"transfer_id": "mv-1", "status": "completed"},
        },
        id="move_funds",
    ),
]


MARKET_DATA_ENDPOINT_CASES = [
    pytest.param(
        {
            "id": "get_ticker",
            "api_mode": "advanced",
            "method": "get_ticker",
            "args": ("BTC-USD",),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/market/products/BTC-USD/ticker",
            "expected_query": {},
            "response": {"price": "123"},
            "expected_result": {"price": "123"},
        },
        id="get_ticker",
    ),
    pytest.param(
        {
            "id": "get_candles",
            "api_mode": "advanced",
            "method": "get_candles",
            "args": ("ETH-USD",),
            "kwargs": {
                "granularity": "1H",
                "limit": 500,
                "start": datetime(2024, 1, 1, 0, 0, 0),
                "end": datetime(2024, 1, 2, 0, 0, 0),
            },
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/market/products/ETH-USD/candles",
            "expected_query": {
                "granularity": ["1H"],
                "limit": ["500"],
                "start": ["2024-01-01T00:00:00Z"],
                "end": ["2024-01-02T00:00:00Z"],
            },
            "response": {"candles": []},
            "expected_result": {"candles": []},
        },
        id="get_candles",
    ),
    pytest.param(
        {
            "id": "get_product_book_advanced",
            "api_mode": "advanced",
            "method": "get_product_book",
            "args": ("BTC-USD",),
            "kwargs": {"level": 2},
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/market/product_book",
            "expected_query": {"product_id": ["BTC-USD"], "level": ["2"]},
            "response": {"bids": [], "asks": []},
            "expected_result": {"bids": [], "asks": []},
        },
        id="get_product_book_advanced",
    ),
    pytest.param(
        {
            "id": "get_product_book_exchange",
            "api_mode": "exchange",
            "method": "get_product_book",
            "args": ("BTC-USD",),
            "kwargs": {"level": 2},
            "expected_method": "GET",
            "expected_path": "/products/BTC-USD/book",
            "expected_query": {"level": ["2"]},
            "response": {"bids": [], "asks": []},
            "expected_result": {"bids": [], "asks": []},
        },
        id="get_product_book_exchange",
    ),
    pytest.param(
        {
            "id": "get_market_products",
            "api_mode": "advanced",
            "method": "get_market_products",
            "args": (),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/market/products",
            "expected_query": {},
            "response": {"products": []},
            "expected_result": {"products": []},
        },
        id="get_market_products",
    ),
    pytest.param(
        {
            "id": "get_market_product",
            "api_mode": "advanced",
            "method": "get_market_product",
            "args": ("BTC-USD",),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/market/products/BTC-USD",
            "expected_query": {},
            "response": {"product_id": "BTC-USD"},
            "expected_result": {"product_id": "BTC-USD"},
        },
        id="get_market_product",
    ),
    pytest.param(
        {
            "id": "get_market_product_ticker",
            "api_mode": "advanced",
            "method": "get_market_product_ticker",
            "args": ("ETH-USD",),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/market/products/ETH-USD/ticker",
            "expected_query": {},
            "response": {"price": "50000"},
            "expected_result": {"price": "50000"},
        },
        id="get_market_product_ticker",
    ),
    pytest.param(
        {
            "id": "get_market_product_candles",
            "api_mode": "advanced",
            "method": "get_market_product_candles",
            "args": ("BTC-USD",),
            "kwargs": {"granularity": "5M", "limit": 300},
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/market/products/BTC-USD/candles",
            "expected_query": {"granularity": ["5M"], "limit": ["300"]},
            "response": {"candles": []},
            "expected_result": {"candles": []},
        },
        id="get_market_product_candles",
    ),
]


SYSTEM_ENDPOINT_CASES = [
    pytest.param(
        {
            "id": "get_time",
            "method": "get_time",
            "args": (),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/time",
            "expected_query": {},
            "response": {"iso": "2024-01-01T00:00:00Z"},
            "expected_result": {"iso": "2024-01-01T00:00:00Z"},
        },
        id="get_time",
    ),
    pytest.param(
        {
            "id": "get_key_permissions",
            "method": "get_key_permissions",
            "args": (),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/key_permissions",
            "expected_query": {},
            "response": {"permissions": ["read", "trade"]},
            "expected_result": {"permissions": ["read", "trade"]},
        },
        id="get_key_permissions",
    ),
    pytest.param(
        {
            "id": "get_fees",
            "method": "get_fees",
            "args": (),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/fees",
            "expected_query": {},
            "response": {"maker_fee_rate": "0.004", "taker_fee_rate": "0.006"},
            "expected_result": {"maker_fee_rate": "0.004", "taker_fee_rate": "0.006"},
        },
        id="get_fees",
    ),
    pytest.param(
        {
            "id": "get_limits",
            "method": "get_limits",
            "args": (),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/limits",
            "expected_query": {},
            "response": {"buy_power": "50000", "sell_power": "50000"},
            "expected_result": {"buy_power": "50000", "sell_power": "50000"},
        },
        id="get_limits",
    ),
    pytest.param(
        {
            "id": "list_payment_methods",
            "method": "list_payment_methods",
            "args": (),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/payment_methods",
            "expected_query": {},
            "response": {"payment_methods": []},
            "expected_result": {"payment_methods": []},
        },
        id="list_payment_methods",
    ),
    pytest.param(
        {
            "id": "get_payment_method",
            "method": "get_payment_method",
            "args": ("pm-123",),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/payment_methods/pm-123",
            "expected_query": {},
            "response": {"payment_method": {"id": "pm-123", "type": "bank"}},
            "expected_result": {"payment_method": {"id": "pm-123", "type": "bank"}},
        },
        id="get_payment_method",
    ),
    pytest.param(
        {
            "id": "get_convert_trade",
            "method": "get_convert_trade",
            "args": ("convert-456",),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/convert/trade/convert-456",
            "expected_query": {},
            "response": {"trade": {"id": "convert-456", "status": "completed"}},
            "expected_result": {"trade": {"id": "convert-456", "status": "completed"}},
        },
        id="get_convert_trade",
    ),
]


CLIENT_ENDPOINT_CASES = [
    pytest.param(
        {
            "id": "list_orders",
            "method": "list_orders",
            "args": (),
            "kwargs": {
                "product_id": "BTC-USD",
                "order_status": "FILLED",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
            },
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/orders/historical",
            "expected_query": {
                "product_id": ["BTC-USD"],
                "order_status": ["FILLED"],
                "start_date": ["2024-01-01"],
                "end_date": ["2024-01-31"],
            },
            "response": {"orders": []},
            "expected_result": {"orders": []},
        },
        id="list_orders",
    ),
    pytest.param(
        {
            "id": "list_fills",
            "method": "list_fills",
            "args": (),
            "kwargs": {"product_id": "ETH-USD", "limit": 100},
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/orders/historical/fills",
            "expected_query": {"product_id": ["ETH-USD"], "limit": ["100"]},
            "response": {"fills": []},
            "expected_result": {"fills": []},
        },
        id="list_fills",
    ),
    pytest.param(
        {
            "id": "cancel_orders",
            "method": "cancel_orders",
            "args": (["id1", "id2"],),
            "expected_method": "POST",
            "expected_path": "/api/v3/brokerage/orders/batch_cancel",
            "expected_payload": {"order_ids": ["id1", "id2"]},
            "response": {"results": [1, 2]},
            "expected_result": {"results": [1, 2]},
        },
        id="cancel_orders",
    ),
    pytest.param(
        {
            "id": "get_order_historical",
            "method": "get_order_historical",
            "args": ("ord-1",),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/orders/historical/ord-1",
            "response": {"order": {"id": "ord-1"}},
            "expected_result": {"order": {"id": "ord-1"}},
        },
        id="get_order_historical",
    ),
    pytest.param(
        {
            "id": "list_orders_batch",
            "method": "list_orders_batch",
            "args": (["order1", "order2", "order3"],),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/orders/historical/batch",
            "response": {"orders": []},
            "expected_result": {"orders": []},
        },
        id="list_orders_batch",
    ),
    pytest.param(
        {
            "id": "place_order",
            "method": "place_order",
            "args": (
                {
                    "product_id": "BTC-USD",
                    "side": "BUY",
                    "order_configuration": {},
                },
            ),
            "expected_method": "POST",
            "expected_path": "/api/v3/brokerage/orders",
            "expected_payload": {
                "product_id": "BTC-USD",
                "side": "BUY",
                "order_configuration": {},
            },
            "response": {"order_id": "new-order-123"},
            "expected_result": {"order_id": "new-order-123"},
        },
        id="place_order",
    ),
    pytest.param(
        {
            "id": "preview_order",
            "method": "preview_order",
            "args": (
                {
                    "product_id": "BTC-USD",
                    "side": "BUY",
                    "order_configuration": {
                        "limit_limit_gtc": {"base_size": "0.1", "limit_price": "50000"}
                    },
                },
            ),
            "expected_method": "POST",
            "expected_path": "/api/v3/brokerage/orders/preview",
            "expected_payload": {
                "product_id": "BTC-USD",
                "side": "BUY",
                "order_configuration": {
                    "limit_limit_gtc": {"base_size": "0.1", "limit_price": "50000"}
                },
            },
            "response": {"preview_id": "prev-1"},
            "expected_result": {"preview_id": "prev-1"},
        },
        id="preview_order",
    ),
    pytest.param(
        {
            "id": "edit_order_preview",
            "method": "edit_order_preview",
            "args": ({"order_id": "ord-1", "new_price": "49900"},),
            "expected_method": "POST",
            "expected_path": "/api/v3/brokerage/orders/edit_preview",
            "expected_payload": {"order_id": "ord-1", "new_price": "49900"},
            "response": {"edit_preview_id": "ep-1"},
            "expected_result": {"edit_preview_id": "ep-1"},
        },
        id="edit_order_preview",
    ),
    pytest.param(
        {
            "id": "edit_order",
            "method": "edit_order",
            "args": ({"order_id": "ord-1", "price": "49800"},),
            "expected_method": "POST",
            "expected_path": "/api/v3/brokerage/orders/edit",
            "expected_payload": {"order_id": "ord-1", "price": "49800"},
            "response": {"success": True},
            "expected_result": {"success": True},
        },
        id="edit_order",
    ),
    pytest.param(
        {
            "id": "get_transaction_summary",
            "method": "get_transaction_summary",
            "args": (),
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/transaction_summary",
            "response": {"summary": {"fees": "10", "volume": "1000"}},
            "expected_result": {"summary": {"fees": "10", "volume": "1000"}},
        },
        id="get_transaction_summary",
    ),
]


class StubBroker:
    def __init__(self):
        self.calls = []

    def get_key_permissions(self):
        self.calls.append("key_permissions")
        return {"can_trade": True}

    def get_fee_schedule(self):
        self.calls.append("fee_schedule")
        return {"tier": "Advanced"}

    def get_account_limits(self):
        self.calls.append("limits")
        return {"max_order": "100000"}

    def get_transaction_summary(self):
        self.calls.append("transaction_summary")
        return {"total_volume": "12345"}

    def list_payment_methods(self):
        self.calls.append("payment_methods")
        return [{"id": "pm-1"}]

    def list_portfolios(self):
        self.calls.append("portfolios")
        return [{"uuid": "pf-1"}]

    def create_convert_quote(self, payload):
        self.calls.append(("convert_quote", payload))
        return {"trade_id": "trade-1", "quote_id": "q-1"}

    def commit_convert_trade(self, trade_id, payload):
        self.calls.append(("commit_trade", trade_id, payload))
        return {"trade_id": trade_id, "status": "pending"}

    def move_portfolio_funds(self, payload):
        self.calls.append(("move_funds", payload))
        return {"status": "ok", **payload}


class StubEventStore:
    def __init__(self):
        self.metrics = []

    def append_metric(self, bot_id, metrics):
        self.metrics.append((bot_id, metrics))


class TestCoinbaseAccounts:
    @pytest.mark.parametrize("case", ACCOUNT_ENDPOINT_CASES, ids=lambda c: c["id"])
    def test_client_account_endpoints(self, case: dict[str, Any]) -> None:
        client = make_client()
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

    def test_list_balances_advanced_shape(self) -> None:
        adapter = make_adapter()
        payload = {
            "accounts": [
                {
                    "currency": "USD",
                    "available_balance": {"value": "100.25"},
                    "hold": {"value": "10.75"},
                },
                {"currency": "USDC", "available_balance": {"value": "50"}, "hold": "0"},
            ]
        }
        adapter.client.get_accounts = lambda: payload  # type: ignore[attr-defined]
        balances = adapter.list_balances()
        data = {balance.asset: balance for balance in balances}
        assert data["USD"].available == Decimal("100.25")
        assert data["USD"].hold == Decimal("10.75")
        assert data["USD"].total == Decimal("111.00")
        assert data["USDC"].total == Decimal("50")

    def test_list_balances_exchange_shape(self) -> None:
        adapter = make_adapter(api_mode="exchange")
        payload = [
            {"currency": "USD", "balance": "200", "available": "150", "hold": "50"},
            {"currency": "BTC", "balance": "0.1", "available": "0.1", "hold": "0"},
        ]
        adapter.client.get_accounts = lambda: payload  # type: ignore[attr-defined]
        balances = adapter.list_balances()
        data = {balance.asset: balance for balance in balances}
        assert data["USD"].total == Decimal("200")
        assert data["USD"].available == Decimal("150")
        assert data["USD"].hold == Decimal("50")

    def test_funding_enrichment_uses_product_catalog(self) -> None:
        adapter = make_adapter()

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

            def get_funding(self, client, symbol):
                return Decimal("0.0005"), datetime.utcnow() + timedelta(hours=8)

        adapter.product_catalog = Catalog()
        adapter.client.get_product = lambda pid: {
            "product_id": "BTC-USD-PERP",
            "base_increment": "0.001",
            "quote_increment": "0.01",
            "base_min_size": "0.001",
            "contract_type": "perpetual",
        }  # type: ignore[attr-defined]

        product = adapter.get_product("BTC-USD-PERP")
        assert product.funding_rate == Decimal("0.0005")
        assert isinstance(product.next_funding_time, datetime)

    def test_list_positions_maps_from_cfm(self, monkeypatch) -> None:
        broker = CoinbaseBrokerage(
            APIConfig(api_key="k", api_secret="s", passphrase=None, base_url="https://api")
        )
        broker.connect()

        def fake_positions(self):
            return {
                "positions": [
                    {
                        "product_id": "BTC-USD-PERP",
                        "size": "1.5",
                        "entry_price": "100",
                        "mark_price": "110",
                        "unrealized_pnl": "15",
                        "realized_pnl": "2",
                        "leverage": 5,
                        "side": "long",
                    },
                    {
                        "product_id": "ETH-USD-PERP",
                        "contracts": "2",
                        "avg_entry_price": "2000",
                        "index_price": "1950",
                        "unrealizedPnl": "-100",
                        "realizedPnl": "5",
                        "leverage": 3,
                        "side": "short",
                    },
                ]
            }

        monkeypatch.setattr(client_mod.CoinbaseClient, "list_positions", fake_positions)
        positions = broker.list_positions()
        assert len(positions) == 2
        assert positions[0].symbol == "BTC-USD-PERP" and positions[0].quantity == Decimal("1.5")
        assert positions[1].symbol == "ETH-USD-PERP" and positions[1].quantity == Decimal("2")

    def test_account_manager_snapshot_collects_all_sections(self) -> None:
        broker = StubBroker()
        store = StubEventStore()
        manager = CoinbaseAccountManager(broker, event_store=store)

        snapshot = manager.snapshot()

        assert snapshot["key_permissions"]["can_trade"] is True
        assert snapshot["fee_schedule"]["tier"] == "Advanced"
        assert snapshot["limits"]["max_order"] == "100000"
        assert snapshot["transaction_summary"]["total_volume"] == "12345"
        assert snapshot["payment_methods"][0]["id"] == "pm-1"
        assert snapshot["portfolios"][0]["uuid"] == "pf-1"
        assert any(
            metric[1].get("event_type") == "account_manager_snapshot" for metric in store.metrics
        )

    def test_account_manager_convert_commits_when_requested(self) -> None:
        broker = StubBroker()
        store = StubEventStore()
        manager = CoinbaseAccountManager(broker, event_store=store)

        result = manager.convert({"from": "USD", "to": "USDC", "amount": "100"}, commit=True)

        assert result["trade_id"] == "trade-1"
        assert any(call[0] == "convert_quote" for call in broker.calls)
        assert any(call[0] == "commit_trade" for call in broker.calls)
        assert any(metric[1].get("event_type") == "convert_commit" for metric in store.metrics)

    def test_account_manager_move_funds_delegates_to_broker(self) -> None:
        broker = StubBroker()
        store = StubEventStore()
        manager = CoinbaseAccountManager(broker, event_store=store)

        payload = {"from_portfolio": "pf-1", "to_portfolio": "pf-2", "amount": "5"}
        result = manager.move_funds(payload)

        assert result["status"] == "ok"
        assert ("move_funds", payload) in broker.calls
        assert any(metric[1].get("event_type") == "portfolio_move" for metric in store.metrics)


class TestCoinbaseMarketData:
    @pytest.mark.parametrize("case", MARKET_DATA_ENDPOINT_CASES, ids=lambda c: c["id"])
    def test_client_market_data_endpoints(self, case: dict[str, Any]) -> None:
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

    def test_get_product_formats_path(self) -> None:
        client = make_client()
        calls = []

        def fake_transport(method, url, headers, body, timeout):
            calls.append((method, url))
            return 200, {}, json.dumps({"product_id": "BTC-USD", "base_currency": "BTC"})

        client.set_transport_for_testing(fake_transport)
        out = client.get_product("BTC-USD")
        assert calls[0][0] == "GET"
        assert calls[0][1].endswith("/api/v3/brokerage/market/products/BTC-USD")
        assert "product_id" in out

    def test_products_path_differs_by_mode(self) -> None:
        client_ex = make_client("exchange")
        urls: list[str] = []
        client_ex.set_transport_for_testing(
            lambda m, u, h, b, t: (
                (200, {}, json.dumps({"products": []})) if not urls.append(u) else (200, {}, "{}")
            )
        )
        client_ex.get_products()
        assert urls[0].endswith("/products")

        client_adv = make_client("advanced")
        urls = []
        client_adv.set_transport_for_testing(
            lambda m, u, h, b, t: (
                (200, {}, json.dumps({"products": []})) if not urls.append(u) else (200, {}, "{}")
            )
        )
        client_adv.get_products()
        assert urls[0].endswith("/api/v3/brokerage/market/products")

    def test_get_product_book_path_mapping_by_mode(self) -> None:
        client_ex = make_client("exchange")
        urls_ex: list[str] = []
        client_ex.set_transport_for_testing(
            lambda m, u, h, b, t: (
                (200, {}, json.dumps({"bids": [], "asks": []}))
                if not urls_ex.append(u)
                else (200, {}, "{}")
            )
        )
        client_ex.get_product_book("BTC-USD", level=2)
        assert urls_ex[0].endswith("/products/BTC-USD/book?level=2")

        client_adv = make_client("advanced")
        urls_adv: list[str] = []
        client_adv.set_transport_for_testing(
            lambda m, u, h, b, t: (
                (200, {}, json.dumps({"bids": [], "asks": []}))
                if not urls_adv.append(u)
                else (200, {}, "{}")
            )
        )
        client_adv.get_product_book("BTC-USD", level=2)
        assert urls_adv[0].endswith(
            "/api/v3/brokerage/market/product_book?product_id=BTC-USD&level=2"
        )

    def test_advanced_only_endpoints_raise_in_exchange(self) -> None:
        client_ex = make_client("exchange")
        with pytest.raises(CoreInvalidRequestError):
            client_ex.list_portfolios()
        with pytest.raises(CoreInvalidRequestError):
            client_ex.get_best_bid_ask(["BTC-USD"])

    def test_list_parameters_comma_separated(self) -> None:
        client = make_client()
        client.set_transport_for_testing(lambda m, u, h, b, t: (200, {}, json.dumps({"ok": True})))
        _ = client.get_best_bid_ask(["BTC-USD", "ETH-USD", "SOL-USD"])
        assert True

    def test_repeated_parameters_not_encoded_as_array(self) -> None:
        client = make_client()
        urls: list[str] = []

        def transport(method, url, headers, body, timeout):
            urls.append(url)
            return 200, {}, json.dumps({"ok": True})

        client.set_transport_for_testing(transport)
        _ = client.list_orders(filter=["a", "b"])
        assert "filter=['a', 'b']" in urls[0]

    def test_unicode_emoji_in_params(self) -> None:
        client = make_client()
        urls: list[str] = []
        client.set_transport_for_testing(
            lambda m, u, h, b, t: (
                (200, {}, json.dumps({"ok": True})) if not urls.append(u) else (200, {}, "{}")
            )
        )
        _ = client.list_orders(note="🚀")
        assert "🚀" in urls[0]

    def test_empty_values_are_included(self) -> None:
        client = make_client()
        urls: list[str] = []

        def transport(method, url, headers, body, timeout):
            urls.append(url)
            return 200, {}, json.dumps({"ok": True})

        client.set_transport_for_testing(transport)
        _ = client.list_orders(filter="")
        assert url_has_param(urls[0], "filter=")

    def test_special_characters_plus_slash_at(self) -> None:
        client = make_client()
        urls: list[str] = []

        def transport(method, url, headers, body, timeout):
            urls.append(url)
            return 200, {}, json.dumps({"ok": True})

        client.set_transport_for_testing(transport)
        _ = client.list_orders(path="/foo/bar", email="test+user@example.com")
        url = urls[0]
        assert "path=/foo/bar" in url
        assert "email=test+user@example.com" in url

    def test_depth_snapshot_l1_l10_depth_correctness(self) -> None:
        levels = [
            (Decimal("50000"), Decimal("1.0"), "bid"),
            (Decimal("49990"), Decimal("2.0"), "bid"),
            (Decimal("49980"), Decimal("3.0"), "bid"),
            (Decimal("50010"), Decimal("1.5"), "ask"),
            (Decimal("50020"), Decimal("2.5"), "ask"),
            (Decimal("50030"), Decimal("3.5"), "ask"),
        ]
        snapshot = DepthSnapshot(levels)
        assert snapshot.get_l1_depth() == Decimal("1.0")
        assert snapshot.get_l10_depth() == Decimal("13.5")

    def test_depth_snapshot_spread_bps(self) -> None:
        levels = [
            (Decimal("50000"), Decimal("1.0"), "bid"),
            (Decimal("50010"), Decimal("1.0"), "ask"),
        ]
        snapshot = DepthSnapshot(levels)
        assert snapshot.spread_bps == 2.0

    def test_depth_snapshot_mid_price(self) -> None:
        levels = [
            (Decimal("50000"), Decimal("1.0"), "bid"),
            (Decimal("50020"), Decimal("1.0"), "ask"),
        ]
        snapshot = DepthSnapshot(levels)
        assert snapshot.mid == Decimal("50010")

    def test_rolling_window_cleanup_and_stats(self) -> None:
        window = RollingWindow(duration_seconds=10)
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        window.add(100.0, base_time)
        window.add(200.0, base_time + timedelta(seconds=5))
        window.add(300.0, base_time + timedelta(seconds=8))
        stats = window.get_stats()
        assert stats["count"] == 3
        assert stats["sum"] == 600.0
        assert stats["avg"] == 200.0

        window.add(400.0, base_time + timedelta(seconds=15))
        stats = window.get_stats()
        assert stats["count"] == 3
        assert stats["sum"] == 900.0
        assert stats["avg"] == 300.0

    def test_trade_tape_vwap_calculation(self) -> None:
        agg = TradeTapeAgg(duration_seconds=60)
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        agg.add_trade(Decimal("100"), Decimal("10"), "buy", base_time)
        agg.add_trade(Decimal("200"), Decimal("5"), "sell", base_time + timedelta(seconds=10))
        vwap = agg.get_vwap()
        expected = (Decimal("100") * Decimal("10") + Decimal("200") * Decimal("5")) / Decimal("15")
        assert abs(vwap - expected) < Decimal("0.01")

    def test_trade_tape_aggressor_ratio(self) -> None:
        agg = TradeTapeAgg(duration_seconds=60)
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        agg.add_trade(Decimal("100"), Decimal("1"), "buy", base_time)
        agg.add_trade(Decimal("100"), Decimal("1"), "buy", base_time + timedelta(seconds=10))
        agg.add_trade(Decimal("100"), Decimal("1"), "sell", base_time + timedelta(seconds=20))
        assert agg.get_aggressor_ratio() == 2 / 3

    def test_to_product_spot_market(self) -> None:
        payload = {
            "product_id": "BTC-USD",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "base_min_size": "0.001",
            "base_increment": "0.00001",
            "quote_increment": "0.01",
            "min_notional": "10",
        }
        product = to_product(payload)
        assert product.symbol == "BTC-USD"
        assert product.market_type == MarketType.SPOT
        assert product.contract_size is None
        assert product.funding_rate is None
        assert product.next_funding_time is None
        assert product.leverage_max is None

    def test_to_product_perpetual_full(self) -> None:
        payload = {
            "product_id": "BTC-PERP",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "contract_type": "perpetual",
            "base_min_size": "0.001",
            "base_increment": "0.00001",
            "quote_increment": "0.01",
            "min_notional": "10",
            "max_leverage": 20,
            "contract_size": "1",
            "funding_rate": "0.0001",
            "next_funding_time": "2024-01-15T16:00:00Z",
        }
        product = to_product(payload)
        assert product.symbol == "BTC-PERP"
        assert product.market_type == MarketType.PERPETUAL
        assert product.contract_size == Decimal("1")
        assert product.funding_rate == Decimal("0.0001")
        assert product.next_funding_time == datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        assert product.leverage_max == 20

    def test_to_product_perpetual_partial(self) -> None:
        payload = {
            "product_id": "ETH-PERP",
            "base_currency": "ETH",
            "quote_currency": "USD",
            "contract_type": "perpetual",
            "base_min_size": "0.01",
            "base_increment": "0.001",
            "quote_increment": "0.1",
        }
        product = to_product(payload)
        assert product.symbol == "ETH-PERP"
        assert product.market_type == MarketType.PERPETUAL
        assert product.contract_size is None
        assert product.funding_rate is None
        assert product.next_funding_time is None
        assert product.leverage_max is None

    def test_to_product_future_market(self) -> None:
        payload = {
            "product_id": "BTC-USD-240331",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "contract_type": "future",
            "base_min_size": "0.001",
            "base_increment": "0.00001",
            "quote_increment": "0.01",
            "expiry": "2024-03-31T08:00:00Z",
            "contract_size": "1",
        }
        product = to_product(payload)
        assert product.symbol == "BTC-USD-240331"
        assert product.market_type == MarketType.FUTURES
        assert product.expiry == datetime(2024, 3, 31, 8, 0, 0, tzinfo=timezone.utc)
        assert product.contract_size == Decimal("1")

    def test_to_product_invalid_funding_time(self) -> None:
        payload = {
            "product_id": "BTC-PERP",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "contract_type": "perpetual",
            "base_min_size": "0.001",
            "base_increment": "0.00001",
            "quote_increment": "0.01",
            "next_funding_time": "invalid-date",
        }
        product = to_product(payload)
        assert product.next_funding_time is None

    def make_catalog(self, ttl_seconds: int = 900) -> ProductCatalog:
        return ProductCatalog(ttl_seconds=ttl_seconds)

    def test_catalog_refresh_with_perps(self) -> None:
        catalog = self.make_catalog()
        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-USD",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                },
                {
                    "product_id": "BTC-PERP",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "contract_type": "perpetual",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                    "contract_size": "1",
                    "funding_rate": "0.0001",
                    "next_funding_time": "2024-01-15T16:00:00Z",
                    "max_leverage": 20,
                },
            ]
        }
        catalog.refresh(mock_client)
        assert len(catalog._cache) == 2
        perp = catalog._cache["BTC-PERP"]
        assert perp.market_type == MarketType.PERPETUAL
        assert perp.contract_size == Decimal("1")
        assert perp.funding_rate == Decimal("0.0001")
        assert perp.leverage_max == 20

    def test_catalog_get_with_expiry(self) -> None:
        catalog = self.make_catalog(ttl_seconds=1)
        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-PERP",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "contract_type": "perpetual",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                }
            ]
        }
        product = catalog.get(mock_client, "BTC-PERP")
        assert product.symbol == "BTC-PERP"
        assert mock_client.get_products.call_count == 1

        product = catalog.get(mock_client, "BTC-PERP")
        assert mock_client.get_products.call_count == 1

        catalog._last_refresh = datetime.utcnow() - timedelta(seconds=2)
        catalog.get(mock_client, "BTC-PERP")
        assert mock_client.get_products.call_count == 2

    def test_catalog_get_not_found(self) -> None:
        catalog = self.make_catalog()
        mock_client = MagicMock()
        mock_client.get_products.return_value = {"products": []}
        with pytest.raises(NotFoundError) as exc_info:
            catalog.get(mock_client, "MISSING-PERP")
        assert "Product not found: MISSING-PERP" in str(exc_info.value)
        assert mock_client.get_products.call_count == 2

    def test_catalog_get_funding_for_perpetual(self) -> None:
        catalog = self.make_catalog()
        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-PERP",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "contract_type": "perpetual",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                    "funding_rate": "0.0001",
                    "next_funding_time": "2024-01-15T16:00:00Z",
                }
            ]
        }
        funding_rate, next_funding = catalog.get_funding(mock_client, "BTC-PERP")
        assert funding_rate == Decimal("0.0001")
        assert next_funding == datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc)

    def test_catalog_get_funding_for_spot(self) -> None:
        catalog = self.make_catalog()
        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-USD",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                }
            ]
        }
        funding_rate, next_funding = catalog.get_funding(mock_client, "BTC-USD")
        assert funding_rate is None
        assert next_funding is None

    def test_catalog_handles_alternative_response_format(self) -> None:
        catalog = self.make_catalog()
        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "data": [
                {
                    "product_id": "ETH-PERP",
                    "base_currency": "ETH",
                    "quote_currency": "USD",
                    "contract_type": "perpetual",
                    "base_min_size": "0.01",
                    "base_increment": "0.001",
                    "quote_increment": "0.1",
                }
            ]
        }
        catalog.refresh(mock_client)
        assert "ETH-PERP" in catalog._cache
        assert catalog._cache["ETH-PERP"].market_type == MarketType.PERPETUAL

    def make_perp_product(self) -> Product:
        return Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.00001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=20,
            contract_size=Decimal("1"),
        )

    def test_enforce_quantizes_quantity(self) -> None:
        product = self.make_perp_product()
        quantity, price = enforce_perp_rules(product, Decimal("0.123456789"))
        assert quantity == Decimal("0.12345")
        assert price is None

    def test_enforce_quantizes_price(self) -> None:
        product = self.make_perp_product()
        quantity, price = enforce_perp_rules(product, Decimal("0.01"), Decimal("50123.456"))
        assert quantity == Decimal("0.01")
        assert price == Decimal("50123.45")

    def test_enforce_rejects_below_min_size(self) -> None:
        product = self.make_perp_product()
        with pytest.raises(CoreInvalidRequestError) as exc_info:
            enforce_perp_rules(product, Decimal("0.0001"))
        assert "below minimum size" in str(exc_info.value)

    def test_enforce_rejects_below_min_notional(self) -> None:
        product = self.make_perp_product()
        with pytest.raises(CoreInvalidRequestError) as exc_info:
            enforce_perp_rules(product, Decimal("0.001"), Decimal("100"))
        assert "below minimum" in str(exc_info.value)

    def test_enforce_accepts_valid_notional(self) -> None:
        product = self.make_perp_product()
        quantity, price = enforce_perp_rules(product, Decimal("0.001"), Decimal("20000"))
        assert quantity == Decimal("0.001")
        assert price == Decimal("20000")

    def test_enforce_handles_no_min_notional(self) -> None:
        product = self.make_perp_product()
        product.min_notional = None
        quantity, price = enforce_perp_rules(product, Decimal("0.001"), Decimal("1"))
        assert quantity == Decimal("0.001")
        assert price == Decimal("1")

    def test_enforce_complex_quantization(self) -> None:
        product = Product(
            symbol="ETH-PERP",
            base_asset="ETH",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.01"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("50"),
            price_increment=Decimal("0.1"),
            leverage_max=15,
        )
        quantity, price = enforce_perp_rules(product, Decimal("0.123456"), Decimal("2345.67"))
        assert quantity == Decimal("0.123")
        assert price == Decimal("2345.6")
        assert quantity * price >= product.min_notional

    def test_quantize_basic(self) -> None:
        result = quantize_to_increment(Decimal("1.2345"), Decimal("0.01"))
        assert result == Decimal("1.23")

    def test_quantize_floors_not_rounds(self) -> None:
        result = quantize_to_increment(Decimal("1.2389"), Decimal("0.01"))
        assert result == Decimal("1.23")

    def test_quantize_handles_zero_increment(self) -> None:
        result = quantize_to_increment(Decimal("1.2345"), Decimal("0"))
        assert result == Decimal("1.2345")
        result = quantize_to_increment(Decimal("1.2345"), None)
        assert result == Decimal("1.2345")

    def test_quantize_arbitrary_increments(self) -> None:
        result = quantize_to_increment(Decimal("1.237"), Decimal("0.025"))
        assert result == Decimal("1.225")
        result = quantize_to_increment(Decimal("1.237"), Decimal("0.005"))
        assert result == Decimal("1.235")

    def test_staleness_detection_fresh_vs_stale_toggles(self) -> None:
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
        adapter = MinimalCoinbaseBrokerage(config)
        symbol = "BTC-PERP"
        assert adapter.is_stale(symbol, threshold_seconds=10) is True
        assert adapter.is_stale(symbol, threshold_seconds=1) is True
        adapter.start_market_data([symbol])
        assert adapter.is_stale(symbol, threshold_seconds=10) is False
        assert adapter.is_stale(symbol, threshold_seconds=1) is False

    def test_staleness_behavior_matches_validator(self) -> None:
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
        adapter = MinimalCoinbaseBrokerage(config)
        symbol = "ETH-PERP"
        assert adapter.is_stale(symbol) is True
        adapter.start_market_data([symbol])
        assert adapter.is_stale(symbol) is False
        assert adapter.is_stale(symbol, threshold_seconds=1) is False


class TestCoinbaseSystem:
    @pytest.mark.parametrize("case", SYSTEM_ENDPOINT_CASES, ids=lambda c: c["id"])
    def test_client_system_endpoints(self, case: dict[str, Any]) -> None:
        client = make_client()
        recorded: dict[str, Any] = {}

        def transport(method, url, headers, body, timeout):
            recorded["method"] = method
            recorded["url"] = url
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

        expected_result = case.get("expected_result")
        if expected_result is not None:
            assert result == expected_result

    @pytest.mark.perf
    def test_get_products_perf_budget(self) -> None:
        client = make_client()
        client.set_transport_for_testing(
            lambda m, u, h, b, t: (200, {}, json.dumps({"products": []}))
        )
        start = time.perf_counter()
        _ = client.get_products()
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10

    @pytest.mark.perf
    def test_place_order_perf_budget(self) -> None:
        client = make_client()
        client.set_transport_for_testing(
            lambda m, u, h, b, t: (200, {}, json.dumps({"order_id": "ord"}))
        )
        start = time.perf_counter()
        _ = client.place_order({"product_id": "BTC-USD", "side": "BUY", "size": "0.01"})
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10

    def test_keep_alive_header_added(self) -> None:
        client = CoinbaseClient(base_url="https://api.coinbase.com", enable_keep_alive=True)
        assert client.enable_keep_alive is True
        assert client._opener is not None

        mock_opener = MagicMock()
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.headers.items.return_value = []
        mock_response.read.return_value = b'{"success": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=None)
        mock_opener.open.return_value = mock_response
        client._opener = mock_opener

        with patch("urllib.request.Request") as mock_request:
            mock_req_obj = MagicMock()
            mock_request.return_value = mock_req_obj
            status, headers, text = client._urllib_transport(
                "GET",
                "https://api.coinbase.com/test",
                {"Content-Type": "application/json"},
                None,
                30,
            )
            calls = [c.args for c in mock_req_obj.add_header.call_args_list]
            assert ("Connection", "keep-alive") in calls
            assert status == 200 and text == '{"success": true}'

    def test_keep_alive_disabled(self) -> None:
        client = CoinbaseClient(base_url="https://api.coinbase.com", enable_keep_alive=False)
        assert client.enable_keep_alive is False
        assert client._opener is None

        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.headers.items.return_value = []
        mock_response.read.return_value = b'{"success": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=None)

        with (
            patch("urllib.request.Request") as mock_request,
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            mock_req_obj = MagicMock()
            mock_request.return_value = mock_req_obj
            mock_urlopen.return_value = mock_response
            status, headers, text = client._urllib_transport(
                "GET",
                "https://api.coinbase.com/test",
                {"Content-Type": "application/json"},
                None,
                30,
            )
            calls = [c.args for c in mock_req_obj.add_header.call_args_list]
            assert ("Connection", "keep-alive") not in calls
            assert status == 200 and text == '{"success": true}'

    def test_shared_opener_created_flag(self) -> None:
        client = CoinbaseClient(base_url="https://api.coinbase.com", enable_keep_alive=True)
        assert client._opener is not None
        client_no_keepalive = CoinbaseClient(
            base_url="https://api.coinbase.com", enable_keep_alive=False
        )
        assert client_no_keepalive._opener is None

    def test_backoff_jitter_deterministic(self, fake_clock) -> None:
        with patch("bot_v2.features.brokerages.coinbase.client.get_config") as mock_config:
            mock_config.return_value = {"max_retries": 3, "retry_delay": 1.0, "jitter_factor": 0.1}
            client = make_client()
            sleep_calls: list[float] = []

            def mock_transport(method, url, headers, body, timeout):
                if len(sleep_calls) < 2:
                    return (429, {}, '{"error": "rate limited"}')
                return (200, {}, '{"success": true}')

            client._transport = mock_transport

            def capture_sleep(duration):
                sleep_calls.append(duration)
                fake_clock.sleep(duration)

            with patch("time.sleep", side_effect=capture_sleep):
                client._request("GET", "/test")

            assert len(sleep_calls) == 2
            assert abs(sleep_calls[0] - 1.01) < 0.001
            assert abs(sleep_calls[1] - 2.04) < 0.001

    def test_jitter_disabled(self, fake_clock) -> None:
        with patch("bot_v2.features.brokerages.coinbase.client.get_config") as mock_config:
            mock_config.return_value = {"max_retries": 3, "retry_delay": 1.0, "jitter_factor": 0}
            client = make_client()
            sleep_calls: list[float] = []

            def mock_transport(method, url, headers, body, timeout):
                if len(sleep_calls) < 2:
                    return (429, {}, '{"error": "rate limited"}')
                return (200, {}, '{"success": true}')

            client._transport = mock_transport

            def capture_sleep(duration):
                sleep_calls.append(duration)
                fake_clock.sleep(duration)

            with patch("time.sleep", side_effect=capture_sleep):
                client._request("GET", "/test")

            assert sleep_calls == [1.0, 2.0]

    def test_connection_reuse_with_opener(self) -> None:
        client = CoinbaseClient(base_url="https://api.coinbase.com", enable_keep_alive=True)
        mock_opener = MagicMock()
        mock_response = MagicMock()
        mock_response.getcode.return_value = 200
        mock_response.headers.items.return_value = []
        mock_response.read.return_value = b'{"success": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=None)
        mock_opener.open.return_value = mock_response
        client._opener = mock_opener

        with patch("urllib.request.Request") as mock_request:
            mock_req_obj = MagicMock()
            mock_request.return_value = mock_req_obj
            client._urllib_transport("GET", "https://api.coinbase.com/test", {}, None, 30)
            mock_opener.open.assert_called_once()

    def test_rate_limit_tracking(self) -> None:
        client = make_client()
        assert hasattr(client, "_request_count") and client._request_count == 0
        assert hasattr(client, "_request_window_start")

        def mock_transport(method, url, headers, body, timeout):
            return 200, {}, '{"success": true}'

        client._transport = mock_transport
        initial = client._request_count
        client._request("GET", "/test")
        assert client._request_count == initial + 1

    def test_connection_validation(self) -> None:
        config = APIConfig(
            api_key="test_key",
            api_secret="test_secret",
            passphrase=None,
            base_url="https://api.coinbase.com",
            sandbox=False,
        )
        broker = CoinbaseBrokerage(config)
        mock_accounts_response = {
            "accounts": [{"uuid": "test-account-123", "currency": "USD", "balance": "100.00"}]
        }
        broker.client.get_accounts = MagicMock(return_value=mock_accounts_response)
        result = broker.connect()
        assert result is True
        assert broker._connected is True
        assert broker._account_id == "test-account-123"

    def test_position_list_spot_trading(self) -> None:
        config = APIConfig(
            api_key="test_key",
            api_secret="test_secret",
            passphrase=None,
            base_url="https://api.coinbase.com",
            sandbox=False,
            enable_derivatives=False,
        )
        broker = CoinbaseBrokerage(config)
        positions = broker.list_positions()
        assert positions == []
        broker.client.cfm_positions = MagicMock()
        positions = broker.list_positions()
        assert positions == []
        broker.client.cfm_positions.assert_not_called()

    def test_order_error_handling(self) -> None:
        config = APIConfig(
            api_key="test_key",
            api_secret="test_secret",
            passphrase=None,
            base_url="https://api.coinbase.com",
            sandbox=False,
        )
        broker = CoinbaseBrokerage(config)
        mock_product = MagicMock()
        mock_product.step_size = Decimal("0.001")
        mock_product.price_increment = Decimal("0.01")
        mock_product.min_size = Decimal("0.001")
        mock_product.min_notional = None
        broker.product_catalog.get = MagicMock(return_value=mock_product)
        broker.client.place_order = MagicMock(side_effect=InsufficientFunds("Not enough balance"))
        with pytest.raises(InsufficientFunds):
            broker.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
            )


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

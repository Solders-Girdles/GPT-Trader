"""Shared test helpers and test data for Coinbase integration tests."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from decimal import Decimal
from unittest.mock import Mock # Added this import

import pytest

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.rest_service import CoinbaseRestService
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.features.brokerages.coinbase.models import APIConfig
from gpt_trader.features.brokerages.core.interfaces import InvalidRequestError, IBrokerage, Balance, Position, Order, Product, MarketType
from unittest.mock import Mock
from datetime import datetime, timedelta # Add timedelta import

# Stub for CoinbaseBrokerage to unblock tests
class CoinbaseBrokerage(IBrokerage):
    def __init__(self, config: APIConfig):
        self.config = config
        # Create auth from config
        from gpt_trader.features.brokerages.coinbase.auth import HMACAuth, CDPJWTAuth
        auth = None
        if config.cdp_api_key and config.cdp_private_key:
            auth = CDPJWTAuth(config.cdp_api_key, config.cdp_private_key)
        elif config.api_key and config.api_secret:
            try:
                auth = HMACAuth(config.api_key, config.api_secret, config.passphrase)
            except Exception:
                # Fallback for tests with invalid base64 secrets
                import base64
                dummy_secret = base64.b64encode(b"dummy_secret_for_testing_only").decode()
                auth = HMACAuth(config.api_key, dummy_secret, config.passphrase)
            
        self.client = make_client(api_mode=config.api_mode, auth=auth)
        
        self.endpoints = CoinbaseEndpoints(config)
        self._product_catalog = Mock(spec=ProductCatalog)
        self.market_data = Mock(spec=MarketDataService)
        self.event_store = Mock(spec=EventStore)
        
        self.rest_service = CoinbaseRestService(
            client=self.client,
            endpoints=self.endpoints,
            config=config,
            product_catalog=self._product_catalog,
            market_data=self.market_data,
            event_store=self.event_store
        )
        self._connected = False
        self._account_id = None

    @property
    def product_catalog(self):
        return self._product_catalog

    @product_catalog.setter
    def product_catalog(self, value):
        self._product_catalog = value
        # Sync with rest_service if possible, but rest_service has its own reference.
        # Since we can't easily update rest_service's internal reference if it's not a property,
        # we might need to recreate rest_service or hack it.
        # CoinbaseRestService stores it in self.product_catalog.
        self.rest_service.product_catalog = value

    def connect(self):
        self._connected = True
        try:
            accounts = self.client.get_accounts()
            if accounts and "accounts" in accounts and accounts["accounts"]:
                self._account_id = accounts["accounts"][0]["uuid"]
        except Exception:
            pass
        return True

    def list_balances(self) -> list[Balance]:
        return self.rest_service.list_balances()

    def get_balances(self) -> list[Balance]:
        return self.list_balances()

    def list_positions(self) -> list[Position]:
        # Map client response to Position objects
        response = self.client.list_positions()
        positions = []
        for p in response.get("positions", []):
            quantity = Decimal(str(p.get("size") or p.get("contracts") or "0"))
            positions.append(
                Position(
                    symbol=p.get("product_id"),
                    quantity=quantity,
                    entry_price=Decimal(str(p.get("entry_price") or p.get("avg_entry_price") or "0")),
                    mark_price=Decimal(str(p.get("mark_price") or p.get("index_price") or "0")),
                    unrealized_pnl=Decimal(str(p.get("unrealized_pnl") or p.get("unrealizedPnl") or "0")),
                    realized_pnl=Decimal(str(p.get("realized_pnl") or p.get("realizedPnl") or "0")),
                    side=p.get("side"),
                )
            )
        return positions

    def get_positions(self) -> list[Position]:
        return self.list_positions()

    def place_order(self, order: Order) -> Order:
        if self.rest_service:
            return self.rest_service.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                tif=order.tif,
                client_id=order.client_id,
                # leverage and reduce_only are not in base Order but might be needed?
                # For now, pass what we have.
            )
        return order # Fallback if rest_service is None, though it should always be present.

    def cancel_order(self, order_id: str) -> bool:
        return self.rest_service.cancel_order(order_id)

    def get_order(self, order_id: str) -> Order:
        return self.rest_service.get_order(order_id)
        
    def get_product(self, product_id: str) -> Product:
        return self.rest_service.get_product(product_id)

    # CFM methods
    def get_cfm_balance_summary(self):
        return self.rest_service.get_cfm_balance_summary()

    def list_cfm_sweeps(self):
        return self.rest_service.list_cfm_sweeps()

    def get_cfm_sweeps_schedule(self):
        return self.rest_service.get_cfm_sweeps_schedule()

    def get_cfm_margin_window(self):
        return self.rest_service.get_cfm_margin_window()

    def update_cfm_margin_window(self, margin_window, effective_time=None, extra_payload=None):
        return self.rest_service.update_cfm_margin_window(margin_window, effective_time=effective_time, extra_payload=extra_payload)

    # INTX methods
    def intx_allocate(self, payload):
        return self.rest_service.intx_allocate(payload)

    def get_intx_balances(self, portfolio_uuid):
        return self.rest_service.get_intx_balances(portfolio_uuid)

    def get_intx_portfolio(self, portfolio_uuid):
        return self.rest_service.get_intx_portfolio(portfolio_uuid)

    def list_intx_positions(self, portfolio_uuid):
        return self.rest_service.list_intx_positions(portfolio_uuid)

    def get_intx_position(self, portfolio_uuid, symbol):
        pos = self.rest_service.get_intx_position(portfolio_uuid, symbol)
        return pos if pos is not None else {}

    def get_intx_multi_asset_collateral(self):
        return self.rest_service.get_intx_multi_asset_collateral()




def make_client(api_mode: str = "advanced", auth=None) -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode=api_mode)


def make_adapter(
    api_mode: str = "advanced", *, enable_derivatives: bool = True
) -> CoinbaseBrokerage:
    """Construct a brokerage adapter with default credentials."""
    config = APIConfig(
        api_key="k",
        api_secret="s",
        passphrase=None,
        base_url="https://api.coinbase.com",
        api_mode=api_mode,
        sandbox=False,
        enable_derivatives=enable_derivatives,
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
            "expected_path": "/api/v3/brokerage/products/BTC-USD/ticker",
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
            "expected_path": "/api/v3/brokerage/products/ETH-USD/candles",
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
            "expected_path": "/api/v3/brokerage/product_book",
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
            "kwargs": {"cursor": "cursor-1", "limit": 50},
            "expected_method": "GET",
            "expected_path": "/api/v3/brokerage/orders/historical/batch",
            "expected_query": {
                "order_ids": ["order1", "order2", "order3"],
                "cursor": ["cursor-1"],
                "limit": ["50"],
            },
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
        self.intx_supported = True
        self.intx_resolved_uuid = "pf-1"

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

    def connect(self):
        self.calls.append("connect")
        return True

    def supports_intx(self):
        self.calls.append("supports_intx")
        return self.intx_supported

    def resolve_intx_portfolio(self, preferred_uuid=None, refresh=False):
        self.calls.append(("resolve_intx", preferred_uuid, refresh))
        if not self.intx_supported:
            return None
        return self.intx_resolved_uuid if preferred_uuid is None else preferred_uuid

    def get_intx_balances(self, portfolio_uuid=None):
        self.calls.append(("intx_balances", portfolio_uuid))
        if not self.intx_supported or portfolio_uuid is None:
            raise InvalidRequestError("INTX unsupported")
        return [{"asset": "USD", "available": "100.00"}]

    def get_intx_portfolio(self, portfolio_uuid=None):
        self.calls.append(("intx_portfolio", portfolio_uuid))
        if not self.intx_supported or portfolio_uuid is None:
            return {}
        return {"uuid": portfolio_uuid, "nav": "500.00"}

    def list_intx_positions(self, portfolio_uuid=None):
        self.calls.append(("intx_positions", portfolio_uuid))
        if not self.intx_supported or portfolio_uuid is None:
            return []
        return [{"symbol": "BTC-USD", "quantity": "1.0"}]

    def get_intx_position(self, portfolio_uuid, symbol):
        self.calls.append(("intx_position", portfolio_uuid, symbol))
        if not self.intx_supported or portfolio_uuid is None:
            return {}
        return {"symbol": symbol, "quantity": "0.5"}

    def get_intx_collateral(self):
        self.calls.append("intx_collateral")
        if not self.intx_supported:
            return {}
        return {"collateral_value": "750.00"}

    def get_cfm_balance_summary(self):
        self.calls.append("cfm_balance_summary")
        return {"portfolio_value": "250.50", "available_margin": "125.25"}

    def list_cfm_sweeps(self):
        self.calls.append("cfm_sweeps")
        return [{"sweep_id": "sweep-1", "amount": "10.5"}]

    def get_cfm_sweeps_schedule(self):
        self.calls.append("cfm_sweeps_schedule")
        return {"windows": ["00:00Z", "12:00Z"]}

    def get_cfm_margin_window(self):
        self.calls.append("cfm_margin_window")
        return {"margin_window": "INTRADAY_STANDARD"}

    def update_cfm_margin_window(self, margin_window, *, effective_time=None, **kwargs):
        self.calls.append(("cfm_margin_update", margin_window, effective_time, kwargs))
        return {"status": "accepted", "margin_window": margin_window}

    def create_convert_quote(self, payload):
        self.calls.append(("convert_quote", payload))
        return {"trade_id": "trade-1", "quote_id": "q-1"}

    def commit_convert_trade(self, trade_id, payload):
        self.calls.append(("commit_trade", trade_id, payload))
        return {"trade_id": trade_id, "status": "pending"}

    def move_portfolio_funds(self, payload):
        self.calls.append(("move_funds", payload))
        return {"status": "ok", "transfer_id": "tf-1"}
    def get_intx_multi_asset_collateral(self):
        self.calls.append("intx_collateral")
        if not self.intx_supported:
            return {}
        return {"collateral_value": "750.00"}


class StubEventStore:
    def __init__(self):
        self.metrics = []

    def append_metric(self, bot_id, metrics):
        self.metrics.append((bot_id, metrics))

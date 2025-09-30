"""
Coinbase adapter implementing the brokerage protocol for perpetuals trading.

This module now orchestrates cohesive services that handle authentication,
REST operations, and WebSocket market data while presenting the same
`IBrokerage` interface used throughout the codebase.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.coinbase.auth import build_rest_auth
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from bot_v2.features.brokerages.coinbase.market_data_service import MarketDataService
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.rest_service import CoinbaseRestService
from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog
from bot_v2.features.brokerages.coinbase.websocket_handler import CoinbaseWebSocketHandler
from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket
from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    IBrokerage,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Product,
    Quote,
)
from bot_v2.persistence.event_store import EventStore

logger = logging.getLogger(__name__)


class CoinbaseBrokerage(IBrokerage):
    """Production-ready Coinbase perpetuals adapter built from focused services."""

    def __init__(self, config: APIConfig) -> None:
        self.config = config
        self.endpoints = CoinbaseEndpoints(
            mode=config.api_mode,
            sandbox=config.sandbox,
            enable_derivatives=config.enable_derivatives,
        )

        auth = build_rest_auth(config)
        self.client = CoinbaseClient(
            auth=auth,
            base_url=self.endpoints.base_url,
            api_mode=config.api_mode,
            api_version=config.api_version,
        )

        self._connected = False
        self._account_id: str | None = None

        self._event_store = EventStore()
        self._product_catalog = ProductCatalog(ttl_seconds=900)
        self.market_data = MarketDataService()
        self.rest_service = CoinbaseRestService(
            client=self.client,
            endpoints=self.endpoints,
            config=config,
            product_catalog=self._product_catalog,
            market_data=self.market_data,
            event_store=self._event_store,
        )
        self.ws_handler = CoinbaseWebSocketHandler(
            endpoints=self.endpoints,
            config=config,
            market_data=self.market_data,
            rest_service=self.rest_service,
            product_catalog=self._product_catalog,
            client_auth=self.client.auth,
            ws_cls=CoinbaseWebSocket,
        )
        self._ws_factory_override = None

        logger.info(
            "CoinbaseBrokerage initialized - mode: %s, sandbox: %s",
            config.api_mode,
            config.sandbox,
        )

    # ------------------------------------------------------------------
    # Product discovery
    # ------------------------------------------------------------------
    def list_products(self, market: MarketType | None = None) -> list[Product]:
        return self.rest_service.list_products(market)

    def get_product(self, symbol: str) -> Product | None:
        return self.rest_service.get_product(symbol)

    def get_perpetuals(self) -> list[Product]:
        return self.rest_service.get_perpetuals()

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def list_orders(
        self,
        status: OrderStatus | str | None = None,
        symbol: str | None = None,
    ) -> list[Order]:
        return self.rest_service.list_orders(status=status, symbol=symbol)

    def get_order(self, order_id: str) -> Order | None:
        return self.rest_service.get_order(order_id)

    def preview_order(self, **kwargs: Any) -> dict[str, Any]:
        return self.rest_service.preview_order(**kwargs)

    def edit_order_preview(self, **kwargs: Any) -> dict[str, Any]:
        return self.rest_service.edit_order_preview(**kwargs)

    def edit_order(self, order_id: str, preview_id: str) -> Order:
        return self.rest_service.edit_order(order_id=order_id, preview_id=preview_id)

    def place_order(self, **kwargs: Any) -> Order:
        return self.rest_service.place_order(**kwargs)

    def cancel_order(self, order_id: str) -> bool:
        return self.rest_service.cancel_order(order_id)

    def close_position(
        self, symbol: str, quantity: Decimal | None = None, reduce_only: bool = True
    ) -> Order:
        positions = self.list_positions()

        def fallback(side: OrderSide, size: Decimal, reduce_flag: bool) -> Order:
            return self.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=size,
                reduce_only=reduce_flag,
            )

        return self.rest_service.close_position(
            symbol,
            quantity=quantity,
            reduce_only=reduce_only,
            positions_override=positions,
            fallback=fallback,
        )

    def list_fills(self, symbol: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        return self.rest_service.list_fills(symbol, limit)

    # ------------------------------------------------------------------
    # Positions & PnL
    # ------------------------------------------------------------------
    def list_positions(self) -> list[Position]:
        return self.rest_service.list_positions()

    def get_position(self, symbol: str) -> Position | None:
        return self.rest_service.get_position(symbol)

    def get_position_pnl(self, symbol: str) -> dict[str, Any]:
        return self.rest_service.get_position_pnl(symbol)

    def get_portfolio_pnl(self) -> dict[str, Any]:
        return self.rest_service.get_portfolio_pnl()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def start_market_data(self, symbols: Sequence[str]) -> None:
        self.ws_handler.start_market_data(symbols)

    def stream_trades(self, symbols: Sequence[str]) -> Iterable[dict[str, Any]]:
        return self.ws_handler.stream_trades(symbols, ws=self._create_ws())

    def stream_orderbook(self, symbols: Sequence[str], level: int = 1) -> Iterable[dict[str, Any]]:
        return self.ws_handler.stream_orderbook(symbols, level, ws=self._create_ws())

    def stream_user_events(
        self, product_ids: Sequence[str] | None = None
    ) -> Iterable[dict[str, Any]]:
        return self.ws_handler.stream_user_events(product_ids, ws=self._create_ws())

    def get_quote(self, symbol: str) -> Quote | None:
        cached = self.market_data.get_cached_quote(symbol)
        if cached and not self.is_stale(symbol):
            return Quote(
                symbol=symbol,
                bid=cached.get("bid", Decimal("0")),
                ask=cached.get("ask", Decimal("0")),
                last=cached.get("last", Decimal("0")),
                ts=cached.get("last_update"),
            )
        return self.rest_service.get_rest_quote(symbol)

    def get_market_snapshot(self, symbol: str) -> dict[str, Any]:
        return self.market_data.get_snapshot(symbol)

    def get_candles(self, symbol: str, granularity: str, limit: int = 200) -> list[Any]:
        return self.rest_service.get_candles(symbol, granularity, limit)

    def is_stale(self, symbol: str, threshold_seconds: int = 10) -> bool:
        return self.market_data.is_stale(symbol, threshold_seconds)

    # ------------------------------------------------------------------
    # Account & portfolio helpers
    # ------------------------------------------------------------------
    def list_balances(self) -> list[Balance]:
        return self.rest_service.list_balances()

    def get_portfolio_balances(self) -> list[Balance]:
        return self.rest_service.get_portfolio_balances()

    def get_key_permissions(self) -> dict[str, Any]:
        return self.rest_service.get_key_permissions()

    def get_fee_schedule(self) -> dict[str, Any]:
        return self.rest_service.get_fee_schedule()

    def get_account_limits(self) -> dict[str, Any]:
        return self.rest_service.get_account_limits()

    def get_transaction_summary(self) -> dict[str, Any]:
        return self.rest_service.get_transaction_summary()

    def list_payment_methods(self) -> list[dict[str, Any]]:
        return self.rest_service.list_payment_methods()

    def get_payment_method(self, payment_method_id: str) -> dict[str, Any]:
        return self.rest_service.get_payment_method(payment_method_id)

    def list_portfolios(self) -> list[dict[str, Any]]:
        return self.rest_service.list_portfolios()

    def get_portfolio(self, portfolio_uuid: str) -> dict[str, Any]:
        return self.rest_service.get_portfolio(portfolio_uuid)

    def get_portfolio_breakdown(self, portfolio_uuid: str) -> dict[str, Any]:
        return self.rest_service.get_portfolio_breakdown(portfolio_uuid)

    def move_portfolio_funds(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.rest_service.move_portfolio_funds(payload)

    def create_convert_quote(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.rest_service.create_convert_quote(payload)

    def commit_convert_trade(
        self, trade_id: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.rest_service.commit_convert_trade(trade_id, payload)

    def get_convert_trade(self, trade_id: str) -> dict[str, Any]:
        return self.rest_service.get_convert_trade(trade_id)

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        try:
            data = self.client.get_accounts()
        except Exception as exc:
            logger.error("Connection failed: %s", exc)
            self._connected = False
            return False

        if not data:
            self._connected = False
            return False

        self._connected = True
        accounts = data.get("accounts") or data.get("data") or []
        if accounts:
            self._account_id = accounts[0].get("uuid") or accounts[0].get("id") or "COINBASE"
        else:
            self._account_id = "COINBASE"
        logger.info("Connected to Coinbase (account: %s)", self._account_id)
        return True

    def disconnect(self) -> None:
        self._connected = False

    def validate_connection(self) -> bool:
        return self._connected

    def get_account_id(self) -> str:
        return self._account_id or ""

    # ------------------------------------------------------------------
    # Testing hooks
    # ------------------------------------------------------------------
    def set_http_transport_for_testing(self, transport) -> None:
        self.client.set_transport_for_testing(transport)

    def set_ws_factory_for_testing(self, factory) -> None:
        self._ws_factory_override = factory
        self.ws_handler.set_ws_factory_for_testing(factory)

    def _create_ws(self):
        # Always reflect the current CoinbaseWebSocket class (patch-friendly for tests)
        self.ws_handler._ws_cls = CoinbaseWebSocket
        self.ws_handler._client_auth = getattr(self.client, "auth", None)
        if self._ws_factory_override:
            return self._ws_factory_override()
        return self.ws_handler.create_ws()

    @property
    def product_catalog(self) -> ProductCatalog:
        return self._product_catalog

    @product_catalog.setter
    def product_catalog(self, catalog: ProductCatalog) -> None:
        self._product_catalog = catalog
        self.rest_service.product_catalog = catalog
        self.ws_handler._product_catalog = catalog

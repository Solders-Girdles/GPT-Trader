from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.coinbase.auth import build_ws_auth_provider
from bot_v2.features.brokerages.coinbase.market_data_service import MarketDataService
from bot_v2.features.brokerages.coinbase.models import APIConfig, normalize_symbol
from bot_v2.features.brokerages.coinbase.rest_service import CoinbaseRestService
from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog
from bot_v2.features.brokerages.coinbase.ws import (
    CoinbaseWebSocket,
    SequenceGuard,
    WSSubscription,
    normalize_market_message,
)

logger = logging.getLogger(__name__)


class CoinbaseWebSocketHandler:
    """Encapsulates Coinbase WebSocket lifecycle and message processing."""

    def __init__(
        self,
        *,
        endpoints,
        config: APIConfig,
        market_data: MarketDataService,
        rest_service: CoinbaseRestService,
        product_catalog: ProductCatalog,
        client_auth: object | None,
        ws_cls: type[CoinbaseWebSocket] = CoinbaseWebSocket,
        metrics_emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._endpoints = endpoints
        self._config = config
        self._market_data = market_data
        self._rest_service = rest_service
        self._product_catalog = product_catalog
        self._client_auth = client_auth
        self._ws_cls = ws_cls
        self._metrics_emitter = metrics_emitter

        self._ws_client: CoinbaseWebSocket | None = None
        self._ws_factory_override = None
        self._sequence_guard = SequenceGuard()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start_market_data(self, symbols: Sequence[str]) -> None:
        if not symbols:
            return
        symbol_list = [normalize_symbol(symbol) for symbol in symbols]
        self._market_data.initialise_symbols(symbol_list)
        ws = self._ensure_ws_client()
        ws.on_message = self._handle_ws_message  # type: ignore[attr-defined]
        subscriptions = [
            WSSubscription(channels=["ticker", "heartbeat"], product_ids=symbol_list),
            WSSubscription(channels=["matches", "heartbeat"], product_ids=symbol_list),
            WSSubscription(channels=["level2", "heartbeat"], product_ids=symbol_list),
        ]
        for sub in subscriptions:
            ws.subscribe(sub)
        logger.info("Started WebSocket market data for %d symbols", len(symbol_list))

    def create_ws(self) -> CoinbaseWebSocket:
        return self._create_ws_instance()

    def stream_trades(
        self, symbols: Sequence[str], ws: CoinbaseWebSocket | None = None
    ) -> Iterable[dict]:
        ws = ws or self._create_ws_instance()
        subscription = WSSubscription(
            channels=["market_trades", "heartbeat"],
            product_ids=[normalize_symbol(symbol) for symbol in symbols],
        )
        ws.subscribe(subscription)
        for message in ws.stream_messages():
            normalised = normalize_market_message(message)
            product_id = normalised.get("product_id") or normalised.get("symbol")
            price = normalised.get("price")
            if isinstance(product_id, str) and isinstance(price, Decimal):
                self._market_data.set_mark(product_id, price)
                self._rest_service.update_position_metrics(product_id)
            yield normalised

    def stream_orderbook(
        self, symbols: Sequence[str], level: int = 1, ws: CoinbaseWebSocket | None = None
    ) -> Iterable[dict]:
        ws = ws or self._create_ws_instance()
        channel = "level2" if level >= 2 else "ticker"
        subscription = WSSubscription(
            channels=[channel, "heartbeat"],
            product_ids=[normalize_symbol(symbol) for symbol in symbols],
        )
        ws.subscribe(subscription)
        for message in ws.stream_messages():
            normalised = normalize_market_message(message)
            product_id = normalised.get("product_id") or normalised.get("symbol")
            if channel == "ticker" and isinstance(product_id, str):
                bid = normalised.get("best_bid") or normalised.get("bid")
                ask = normalised.get("best_ask") or normalised.get("ask")
                if isinstance(bid, Decimal) and isinstance(ask, Decimal):
                    mid = (bid + ask) / 2
                    self._market_data.set_mark(product_id, mid)
                    self._rest_service.update_position_metrics(product_id)
            yield normalised

    def stream_user_events(
        self, product_ids: Sequence[str] | None = None, ws: CoinbaseWebSocket | None = None
    ) -> Iterable[dict]:
        ws = ws or self._create_ws_instance()
        subscription = WSSubscription(
            channels=["user", "heartbeat"],
            product_ids=[normalize_symbol(symbol) for symbol in (product_ids or [])],
        )
        ws.subscribe(subscription)
        for message in ws.stream_messages():
            annotated = self._sequence_guard.annotate(message)
            event_type = annotated.get("type") or annotated.get("event_type")
            if event_type == "fill":
                self._rest_service.process_fill_for_pnl(annotated)
            yield annotated

    def set_ws_factory_for_testing(self, factory) -> None:
        self._ws_factory_override = factory
        if self._ws_client is not None:
            self._ws_client = None

    def set_metrics_emitter(self, emitter: Callable[[dict[str, Any]], None] | None) -> None:
        """Attach a streaming metrics emitter for WebSocket events."""

        self._metrics_emitter = emitter
        if self._ws_client is not None and hasattr(self._ws_client, "set_metrics_emitter"):
            self._ws_client.set_metrics_emitter(emitter)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_ws_client(self) -> CoinbaseWebSocket:
        if self._ws_client is None:
            self._ws_client = self._create_ws_instance()
        return self._ws_client

    def _create_ws_instance(self) -> CoinbaseWebSocket:
        if self._ws_factory_override:
            return self._ws_factory_override()
        auth_provider = build_ws_auth_provider(self._config, self._client_auth)
        return self._ws_cls(
            url=self._endpoints.websocket_url(),
            ws_auth_provider=auth_provider,
            metrics_emitter=self._metrics_emitter,
        )

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------
    def _handle_ws_message(self, message: dict) -> None:
        try:
            normalised = self._normalize_ws_message(message)
            if not normalised:
                return
            channel = normalised.get("type")
            symbol = normalised.get("product_id")
            if not symbol or not self._market_data.has_symbol(symbol):
                return
            now = datetime.utcnow()
            if channel == "ticker":
                bid_raw = normalised.get("best_bid") or normalised.get("bid")
                ask_raw = normalised.get("best_ask") or normalised.get("ask")
                last_raw = normalised.get("price") or normalised.get("last")
                bid = Decimal(str(bid_raw)) if bid_raw is not None else None
                ask = Decimal(str(ask_raw)) if ask_raw is not None else None
                last = Decimal(str(last_raw)) if last_raw is not None else None
                self._market_data.update_ticker(symbol, bid, ask, last, now)
            elif channel == "match":
                size_raw = normalised.get("size")
                if size_raw is not None:
                    size = Decimal(str(size_raw))
                    self._market_data.record_trade(symbol, size, now)
            elif channel == "l2update":
                changes = normalised.get("changes", [])
                self._market_data.update_depth(symbol, changes)
        except Exception as exc:
            logger.error("Error handling WebSocket message: %s", exc, exc_info=True)

    def _normalize_ws_message(self, message: dict) -> dict | None:
        if not message:
            return None
        channel = message.get("type", message.get("channel", message.get("event", "")))
        channel_map = {
            "ticker": ["ticker", "tickers", "ticker_batch", "tick"],
            "match": ["match", "matches", "trade", "trades", "executed_trade"],
            "l2update": ["l2update", "level2", "l2", "level2_batch", "orderbook", "book"],
        }
        normalised_channel = None
        for canonical, variants in channel_map.items():
            if any(variant in channel.lower() for variant in variants):
                normalised_channel = canonical
                break
        if not normalised_channel:
            return None
        message["type"] = normalised_channel
        if "product_id" not in message:
            message["product_id"] = message.get("symbol", message.get("instrument", ""))
        if normalised_channel == "ticker":
            if "price" not in message:
                message["price"] = message.get(
                    "last_price", message.get("last", message.get("close"))
                )
            if "best_bid" not in message:
                message["best_bid"] = message.get("bid", message.get("bid_price", "0"))
            if "best_ask" not in message:
                message["best_ask"] = message.get("ask", message.get("ask_price", "0"))
        elif normalised_channel == "match":
            if "size" not in message:
                message["size"] = message.get("quantity", message.get("amount", "0"))
            if "price" not in message:
                message["price"] = message.get("trade_price", message.get("execution_price", "0"))
        elif normalised_channel == "l2update":
            if "changes" not in message:
                if "updates" in message:
                    message["changes"] = message["updates"]
                elif "bids" in message and "asks" in message:
                    changes = []
                    for bid in message.get("bids", [])[:10]:
                        if len(bid) >= 2:
                            changes.append(["buy", str(bid[0]), str(bid[1])])
                    for ask in message.get("asks", [])[:10]:
                        if len(ask) >= 2:
                            changes.append(["sell", str(ask[0]), str(ask[1])])
                    message["changes"] = changes
                elif "data" in message and isinstance(message["data"], list):
                    message["changes"] = message["data"]
        return message

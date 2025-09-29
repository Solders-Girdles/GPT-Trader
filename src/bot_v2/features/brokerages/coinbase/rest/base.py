"""Shared state and helpers for Coinbase REST service mixins."""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from .....errors import ValidationError
from .....persistence.event_store import EventStore
from ...core.interfaces import (
    InsufficientFunds,
    InvalidRequestError,
    NotFoundError,
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)
from ..client import CoinbaseClient
from ..endpoints import CoinbaseEndpoints
from ..market_data_service import MarketDataService
from ..models import APIConfig, normalize_symbol, to_order
from ..utilities import FundingCalculator, PositionState, ProductCatalog, quantize_to_increment

logger = logging.getLogger(__name__)


class CoinbaseRestServiceBase:
    """Holds shared collaborators and internal helpers."""

    def __init__(
        self,
        *,
        client: CoinbaseClient,
        endpoints: CoinbaseEndpoints,
        config: APIConfig,
        product_catalog: ProductCatalog,
        market_data: MarketDataService,
        event_store: EventStore,
    ) -> None:
        self.client = client
        self.endpoints = endpoints
        self.config = config
        self.product_catalog = product_catalog
        self.market_data = market_data
        self._event_store = event_store
        self._funding_calculator = FundingCalculator()
        self._positions: dict[str, PositionState] = {}

    # ------------------------------------------------------------------
    # Order payload helpers
    # ------------------------------------------------------------------
    def _build_order_payload(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None,
        stop_price: Decimal | None,
        tif: TimeInForce,
        client_id: str | None,
        reduce_only: bool | None,
        leverage: int | None,
        post_only: bool = False,
        include_client_id: bool = True,
    ) -> dict[str, object]:
        pid = normalize_symbol(symbol)
        try:
            product = self.product_catalog.get(self.client, pid)
        except NotFoundError:
            product = self.get_product(pid)
            if product is None:
                raise

        def _coerce_enum(enum_cls: Any, raw: Any, field: str) -> Any:
            if isinstance(raw, enum_cls):
                return raw
            if raw is None:
                raise ValidationError(f"{field} is required", field=field)
            token = str(raw)
            try:
                return enum_cls[token.upper()]
            except (KeyError, AttributeError):
                for member in enum_cls:  # type: ignore[arg-type]
                    if token.lower() == str(member.value).lower():
                        return member
            raise ValidationError(f"Unsupported {field}: {token}", field=field)

        side_enum = _coerce_enum(OrderSide, side, "side")
        order_type_enum = _coerce_enum(OrderType, order_type, "order_type")

        adjusted_quantity = quantize_to_increment(quantity, product.step_size)

        adjusted_price = price
        if order_type_enum in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if adjusted_price is None:
                raise ValidationError("price is required for limit orders", field="price")
            adjusted_price = quantize_to_increment(adjusted_price, product.price_increment)

        adjusted_stop = stop_price
        if adjusted_stop is not None:
            adjusted_stop = quantize_to_increment(adjusted_stop, product.price_increment)

        if adjusted_quantity < product.min_size:
            raise ValidationError(
                f"quantity {adjusted_quantity} is below min_size {product.min_size}",
                field="quantity",
                value=str(adjusted_quantity),
            )

        tif_enum = (
            tif if isinstance(tif, TimeInForce) else _coerce_enum(TimeInForce, tif, "time_in_force")
        )
        gtd_requested = tif_enum not in TimeInForce and str(tif).upper() in {
            "GTD",
            "GOOD_TILL_DATE",
        }
        if gtd_requested:
            tif_enum = TimeInForce.GTC

        payload: dict[str, object] = {
            "product_id": pid,
            "side": side_enum.value.upper(),
        }

        if product.min_notional:
            reference_price = adjusted_price
            if reference_price is None:
                rest_quote = self.get_rest_quote(pid)
                reference_price = rest_quote.last if rest_quote else Decimal("0")
            notional = adjusted_quantity * (reference_price or Decimal("0"))
            if notional < product.min_notional:
                raise ValidationError(
                    f"notional {notional} below min_notional {product.min_notional}",
                    field="quantity",
                    value=str(adjusted_quantity),
                )

        if order_type_enum == OrderType.LIMIT:
            limit_map = {
                TimeInForce.IOC: "limit_limit_ioc",
                TimeInForce.FOK: "limit_limit_fok",
            }
            config_key = limit_map.get(tif_enum, "limit_limit_gtc")
            payload["order_configuration"] = {
                config_key: {
                    "base_size": str(adjusted_quantity),
                    "limit_price": str(adjusted_price or Decimal("0")),
                }
            }
        elif order_type_enum == OrderType.MARKET:
            payload["order_configuration"] = {
                "market_market_ioc": {"base_size": str(adjusted_quantity)}
            }
        elif order_type_enum == OrderType.STOP_LIMIT and adjusted_stop is not None:
            payload["order_configuration"] = {
                "stop_limit_stop_limit_gtc": {
                    "base_size": str(adjusted_quantity),
                    "limit_price": str(adjusted_price or Decimal("0")),
                    "stop_price": str(adjusted_stop),
                }
            }
        else:
            payload["type"] = order_type_enum.value
            payload["size"] = str(adjusted_quantity)
            payload["time_in_force"] = tif_enum.value
            if adjusted_price is not None:
                payload["price"] = str(adjusted_price)
            if adjusted_stop is not None:
                payload["stop_price"] = str(adjusted_stop)

        if "order_configuration" in payload:
            payload.setdefault("type", order_type_enum.value)
            payload.setdefault("size", str(adjusted_quantity))
            payload.setdefault("time_in_force", tif_enum.value)
            if adjusted_price is not None:
                payload.setdefault("price", str(adjusted_price))
            if adjusted_stop is not None:
                payload.setdefault("stop_price", str(adjusted_stop))

        if reduce_only is not None:
            payload["reduce_only"] = reduce_only
        if leverage is not None:
            payload["leverage"] = leverage
        if post_only:
            payload["post_only"] = True
        if include_client_id:
            payload["client_order_id"] = client_id or f"perps_{uuid.uuid4().hex[:12]}"
        payload["type"] = payload.get("type", order_type_enum.value)
        payload["size"] = str(adjusted_quantity)
        payload["quantity"] = str(adjusted_quantity)
        return payload

    def _execute_order_payload(
        self,
        symbol: str,
        payload: dict[str, object],
        client_id: str | None,
    ) -> Order:
        if os.getenv("ORDER_PREVIEW_ENABLED"):
            try:
                self.client.preview_order(payload)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - preview optional
                logger.debug("Order preview failed (ignored): %s", exc)
        try:
            data = self.client.place_order(payload)
            return to_order(data or {})
        except InsufficientFunds as exc:
            logger.error("Insufficient funds for %s: %s", payload.get("product_id"), exc)
            raise
        except ValidationError as exc:
            logger.error("Order validation failed for %s: %s", payload.get("product_id"), exc)
            raise
        except InvalidRequestError as exc:
            if client_id and "duplicate" in str(exc).lower():
                existing = self._find_existing_order_by_client_id(
                    payload.get("product_id", ""), client_id
                )
                if existing:
                    logger.info(
                        "Resolved duplicate client_order_id for %s via order %s",
                        payload.get("product_id"),
                        existing.id,
                    )
                    return existing
            logger.error("Order placement failed for %s: %s", payload.get("product_id"), exc)
            raise
        except Exception as exc:
            logger.error(
                "Order placement failed for %s: %s: %s",
                payload.get("product_id"),
                exc.__class__.__name__,
                exc,
            )
            raise

    def _find_existing_order_by_client_id(self, product_id: str, client_id: str) -> Order | None:
        if not client_id:
            return None
        try:
            data = self.client.list_orders(product_id=product_id) or {}
        except Exception as exc:
            logger.debug("Could not resolve duplicate order for %s: %s", product_id, exc)
            return None
        items = data.get("orders") or data.get("data") or []
        matches = [item for item in items if str(item.get("client_order_id")) == client_id]
        if not matches:
            return None

        def _timestamp(order: dict[str, Any]) -> datetime:
            raw = order.get("created_at") or order.get("submitted_at")
            if not raw:
                return datetime.min
            try:
                normalized = str(raw).replace("Z", "+00:00")
                return datetime.fromisoformat(normalized)
            except Exception:
                return datetime.min

        best = max(matches, key=_timestamp)
        return to_order(best)

    # ------------------------------------------------------------------
    # Position bookkeeping helpers
    # ------------------------------------------------------------------
    def _update_position_metrics(self, symbol: str) -> None:
        if symbol not in self._positions:
            return
        mark = self.market_data.get_mark(symbol)
        if mark is None:
            return
        position = self._positions[symbol]
        unrealized = position.get_unrealized_pnl(mark)
        funding_rate, next_funding_time = self.product_catalog.get_funding(self.client, symbol)
        funding_delta = self._funding_calculator.accrue_if_due(
            symbol=symbol,
            position_size=position.quantity,
            position_side=position.side,
            mark_price=mark,
            funding_rate=funding_rate,
            next_funding_time=next_funding_time,
        )
        if funding_delta != 0:
            position.realized_pnl += funding_delta
            self._event_store.append_metric(
                bot_id="coinbase_perps",
                metrics={
                    "type": "funding",
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "side": position.side,
                    "quantity": str(position.quantity),
                    "funding_rate": str(funding_rate or Decimal("0")),
                    "mark_price": str(mark),
                    "funding_amount": str(funding_delta),
                },
            )
        position_value = position.quantity * mark
        self._event_store.append_position(
            bot_id="coinbase_perps",
            position={
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": position.side,
                "quantity": str(position.quantity),
                "entry_price": str(position.entry_price),
                "mark_price": str(mark),
                "unrealized_pnl": str(unrealized),
                "realized_pnl": str(position.realized_pnl),
                "position_value": str(position_value),
            },
        )

    @property
    def positions(self) -> dict[str, PositionState]:
        return self._positions

    def update_position_metrics(self, symbol: str) -> None:
        self._update_position_metrics(symbol)


__all__ = ["CoinbaseRestServiceBase", "logger"]

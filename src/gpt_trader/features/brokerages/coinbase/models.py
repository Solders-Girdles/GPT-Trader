"""
Models and mappers specific to Coinbase responses (scaffold).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from gpt_trader.core import (
    Candle,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Product,
    Quote,
    TimeInForce,
)
from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.quantities import quantity_from

logger = get_logger(__name__, component="coinbase_models")


@dataclass
class APIConfig:
    api_key: str
    api_secret: str
    passphrase: str | None
    base_url: str
    sandbox: bool = False
    ws_url: str | None = None
    enable_derivatives: bool = True
    # CDP (Coinbase Developer Platform) JWT auth fields
    cdp_api_key: str | None = None  # API key name from CDP
    cdp_private_key: str | None = None  # EC private key in PEM format
    auth_type: str = "JWT"  # JWT only (legacy HMAC removed)
    api_version: str = "2024-10-24"  # CB-VERSION header value (format: YYYY-MM-DD)
    # API mode determines endpoint paths and auth requirements
    api_mode: Literal["advanced", "exchange"] = (
        "advanced"  # "advanced" for AT v3, "exchange" for legacy
    )


def normalize_symbol(symbol: str) -> str:
    """Normalize Coinbase product id to internal form.

    Coinbase typically uses symbols like 'BTC-USD'. Derivatives may include suffixes.
    This helper is a placeholder for any mapping needs.
    """
    return symbol.upper()


def to_product(payload: dict) -> Product:
    market_type = MarketType.SPOT
    if payload.get("contract_type") in {"future", "perpetual"}:
        market_type = (
            MarketType.PERPETUAL
            if payload.get("contract_type") == "perpetual"
            else MarketType.FUTURES
        )

    # Parse funding time if available
    next_funding_time = None
    if payload.get("next_funding_time"):
        try:
            next_funding_time = datetime.fromisoformat(payload["next_funding_time"])
        except (ValueError, TypeError) as exc:
            logger.error(
                "Failed to parse next_funding_time",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="to_product",
                symbol=payload.get("product_id", "unknown"),
            )

    return Product(
        symbol=normalize_symbol(payload.get("product_id") or payload.get("id") or ""),
        base_asset=payload.get("base_currency") or payload.get("base_asset") or "",
        quote_asset=payload.get("quote_currency") or payload.get("quote_asset") or "",
        market_type=market_type,
        min_size=Decimal(str(payload.get("base_min_size") or payload.get("min_size") or "0")),
        step_size=Decimal(
            str(payload.get("base_increment") or payload.get("step_size") or "0.00000001")
        ),
        min_notional=(
            Decimal(str(payload.get("min_notional") or "0"))
            if payload.get("min_notional")
            else None
        ),
        price_increment=Decimal(
            str(payload.get("quote_increment") or payload.get("price_increment") or "0.01")
        ),
        leverage_max=int(payload.get("max_leverage", 1)) if payload.get("max_leverage") else None,
        expiry=datetime.fromisoformat(payload["expiry"]) if payload.get("expiry") else None,
        # Perpetuals-specific fields
        contract_size=(
            Decimal(str(payload.get("contract_size", "1")))
            if payload.get("contract_size")
            else None
        ),
        funding_rate=(
            Decimal(str(payload.get("funding_rate", "0"))) if payload.get("funding_rate") else None
        ),
        next_funding_time=next_funding_time,
    )


def to_quote(payload: dict) -> Quote:
    symbol = normalize_symbol(payload.get("product_id") or payload.get("symbol") or "")
    bid = Decimal(str(payload.get("best_bid") or payload.get("bid") or "0"))
    ask = Decimal(str(payload.get("best_ask") or payload.get("ask") or "0"))

    last_raw = payload.get("price") or payload.get("last") or "0"
    if (last_raw in (None, "", 0, "0")) and payload.get("trades"):
        try:
            last_raw = payload["trades"][0].get("price") or last_raw
        except Exception as exc:
            logger.error(
                "Failed to extract trade price from quote payload",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="to_quote",
                symbol=symbol,
            )
    last = Decimal(str(last_raw or "0"))

    ts_raw = payload.get("time") or payload.get("ts")
    if not ts_raw and payload.get("trades"):
        try:
            ts_raw = payload["trades"][0].get("time")
        except Exception as exc:
            logger.error(
                "Failed to extract trade time from quote payload",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="to_quote",
                symbol=symbol,
            )
            ts_raw = None
    ts = datetime.fromisoformat(
        ts_raw.replace("Z", "+00:00") if isinstance(ts_raw, str) else datetime.utcnow().isoformat()
    )

    return Quote(symbol=symbol, bid=bid, ask=ask, last=last, ts=ts)


def to_candle(payload: dict) -> Candle:
    ts_str = payload.get("time") or payload.get("ts") or datetime.utcnow().isoformat()
    return Candle(
        ts=datetime.fromisoformat(ts_str),
        open=Decimal(str(payload.get("open"))),
        high=Decimal(str(payload.get("high"))),
        low=Decimal(str(payload.get("low"))),
        close=Decimal(str(payload.get("close"))),
        volume=Decimal(str(payload.get("volume"))),
    )


_STATUS_MAP = {
    "pending": OrderStatus.PENDING,
    "open": OrderStatus.SUBMITTED,
    "new": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
}


def to_order(payload: dict) -> Order:
    status = _STATUS_MAP.get(str(payload.get("status", "")).lower(), OrderStatus.SUBMITTED)
    side = OrderSide.BUY if str(payload.get("side", "")).lower() == "buy" else OrderSide.SELL
    otype_str = str(payload.get("type", "")).lower()
    if otype_str in ("limit",):
        otype = OrderType.LIMIT
    elif otype_str in ("market",):
        otype = OrderType.MARKET
    elif otype_str in ("stop", "stop_market"):
        otype = OrderType.STOP
    elif otype_str in ("stop_limit",):
        otype = OrderType.STOP_LIMIT
    else:
        otype = OrderType.LIMIT

    tif = TimeInForce.GTC
    tif_str = str(payload.get("time_in_force", "")).lower()
    if tif_str == "ioc":
        tif = TimeInForce.IOC
    elif tif_str == "fok":
        tif = TimeInForce.FOK

    submitted = payload.get("created_at") or payload.get("submitted_at")
    updated = payload.get("updated_at") or submitted

    raw_quantity = quantity_from(payload, default=None)
    if raw_quantity is None:
        fallback_size = (
            payload.get("size")
            or payload.get("contracts")
            or payload.get("position_quantity")
            or "0"
        )
        raw_quantity = Decimal(str(fallback_size))
    quantity = Decimal(str(raw_quantity))

    filled_raw = (
        payload.get("filled_quantity")
        or payload.get("filled_size")
        or payload.get("filled_quantity")
        or 0
    )
    filled_quantity = Decimal(str(filled_raw))

    order = Order(
        id=str(payload.get("order_id") or payload.get("id") or ""),
        client_id=payload.get("client_order_id") or payload.get("client_id"),
        symbol=normalize_symbol(payload.get("product_id") or payload.get("symbol") or ""),
        side=side,
        type=otype,
        quantity=quantity,
        price=Decimal(str(payload.get("price"))) if payload.get("price") else None,
        stop_price=Decimal(str(payload.get("stop_price"))) if payload.get("stop_price") else None,
        tif=tif,
        status=status,
        filled_quantity=filled_quantity,
        avg_fill_price=(
            Decimal(str(payload.get("average_filled_price") or payload.get("avg_fill_price") or 0))
            if payload.get("average_filled_price") or payload.get("avg_fill_price")
            else None
        ),
        submitted_at=datetime.fromisoformat(submitted) if submitted else datetime.utcnow(),
        updated_at=datetime.fromisoformat(updated) if updated else datetime.utcnow(),
    )

    return order


def to_position(payload: dict) -> Position:
    # Derive side and quantity
    position_quantity = quantity_from(payload, default=None)
    if position_quantity is None:
        fallback_size = (
            payload.get("size")
            or payload.get("position_quantity")
            or payload.get("contracts")
            or "0"
        )
        position_quantity = Decimal(str(fallback_size))
    quantity = Decimal(str(position_quantity))
    raw_side = str(payload.get("side") or ("long" if quantity >= 0 else "short"))
    side = "long" if raw_side.lower().startswith("l") else "short"

    entry = payload.get("entry_price") or payload.get("avg_entry_price") or 0
    mark = payload.get("mark_price") or payload.get("index_price") or payload.get("last") or 0
    upnl = payload.get("unrealized_pnl") or payload.get("unrealizedPnl") or 0
    rpnl = payload.get("realized_pnl") or payload.get("realizedPnl") or 0
    lev = payload.get("leverage") or payload.get("max_leverage")

    position = Position(
        symbol=normalize_symbol(payload.get("product_id") or payload.get("symbol") or ""),
        quantity=Decimal(str(abs(quantity))),
        entry_price=Decimal(str(entry)),
        mark_price=Decimal(str(mark)),
        unrealized_pnl=Decimal(str(upnl)),
        realized_pnl=Decimal(str(rpnl)),
        leverage=int(lev) if lev is not None else None,
        side=side,
    )

    return position

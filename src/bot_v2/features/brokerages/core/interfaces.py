"""
Brokerage core interfaces and shared types for spot and derivatives.

These interfaces define a stable contract for broker adapters (e.g., Coinbase),
allowing orchestration and strategies to remain broker-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Literal, Protocol


class MarketType(Enum):
    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Balance:
    asset: str
    total: Decimal
    available: Decimal
    hold: Decimal


@dataclass
class Product:
    symbol: str  # normalized like BTC-USD or BTC-USD-PERP
    base_asset: str
    quote_asset: str
    market_type: MarketType
    min_size: Decimal
    step_size: Decimal
    min_notional: Decimal | None
    price_increment: Decimal
    leverage_max: int | None = None
    expiry: datetime | None = None
    # Perpetuals-specific fields
    contract_size: Decimal | None = None  # Size of one contract in base asset
    funding_rate: Decimal | None = None  # Current funding rate (if available)
    next_funding_time: datetime | None = None  # Next funding timestamp


@dataclass
class Quote:
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    ts: datetime


@dataclass
class Candle:
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(init=False)
class Order:
    id: str
    client_id: str | None
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Decimal
    price: Decimal | None
    stop_price: Decimal | None
    tif: TimeInForce
    status: OrderStatus
    filled_quantity: Decimal
    avg_fill_price: Decimal | None
    submitted_at: datetime
    updated_at: datetime

    def __init__(
        self,
        *,
        id: str,
        client_id: str | None,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        tif: TimeInForce,
        status: OrderStatus,
        submitted_at: datetime,
        updated_at: datetime,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        avg_fill_price: Decimal | None = None,
        quantity: Decimal,
        filled_quantity: Decimal | None = None,
    ) -> None:
        self.id = id
        self.client_id = client_id
        self.symbol = symbol
        self.side = side
        self.type = type
        self.price = price
        self.stop_price = stop_price
        self.tif = tif
        self.status = status
        self.avg_fill_price = avg_fill_price
        self.submitted_at = submitted_at
        self.updated_at = updated_at
        base_quantity = Decimal(str(quantity))
        base_filled = Decimal(str(filled_quantity)) if filled_quantity is not None else Decimal("0")

        self.quantity = base_quantity
        self.filled_quantity = base_filled


@dataclass(init=False)
class Position:
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    leverage: int | None
    side: Literal["long", "short"]

    def __init__(
        self,
        *args: Any,
        symbol: str | None = None,
        entry_price: Decimal | None = None,
        mark_price: Decimal | None = None,
        unrealized_pnl: Decimal | None = None,
        realized_pnl: Decimal | None = None,
        leverage: int | None = None,
        side: Literal["long", "short"] | None = None,
        quantity: Decimal | None = None,
    ) -> None:
        base_quantity = quantity

        if args:
            if len(args) != 8:
                raise TypeError(
                    "Position positional init expects 8 arguments: symbol, quantity, entry_price, mark_price,"
                    " unrealized_pnl, realized_pnl, leverage, side"
                )
            (
                symbol,
                legacy_quantity,
                entry_price,
                mark_price,
                unrealized_pnl,
                realized_pnl,
                leverage,
                side,
            ) = args
            base_quantity = legacy_quantity if base_quantity is None else base_quantity

        if (
            symbol is None
            or entry_price is None
            or mark_price is None
            or unrealized_pnl is None
            or realized_pnl is None
            or side is None
        ):
            raise TypeError(
                "Position requires symbol, entry_price, mark_price, unrealized_pnl, realized_pnl, and side"
            )

        if base_quantity is None:
            raise TypeError("Position requires 'quantity'")

        self.symbol = str(symbol)
        self.quantity = Decimal(str(base_quantity))
        self.entry_price = Decimal(str(entry_price))
        self.mark_price = Decimal(str(mark_price))
        self.unrealized_pnl = Decimal(str(unrealized_pnl))
        self.realized_pnl = Decimal(str(realized_pnl))
        self.leverage = leverage
        self.side = side


class BrokerageError(Exception):
    """Base error for brokerage adapters."""


class RateLimitError(BrokerageError):
    pass


class AuthError(BrokerageError):
    pass


class NotFoundError(BrokerageError):
    pass


class InvalidRequestError(BrokerageError):
    pass


class InsufficientFunds(BrokerageError):
    pass


class PermissionDeniedError(BrokerageError):
    pass


@dataclass
class BrokerHealth:
    """Health status of brokerage connection."""
    connected: bool
    api_responsive: bool
    last_check_timestamp: float
    error_message: str | None = None


class IBrokerage(Protocol):
    """Brokerage protocol supporting spot and derivatives."""

    # Connectivity
    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def validate_connection(self) -> bool: ...
    def check_health(self) -> BrokerHealth: ...

    # Accounts and balances
    def get_account_id(self) -> str: ...
    def list_balances(self) -> list[Balance]: ...

    # Products and market data
    def list_products(self, market: MarketType | None = None) -> list[Product]: ...
    def get_product(self, symbol: str) -> Product: ...
    def get_quote(self, symbol: str) -> Quote: ...
    def get_candles(self, symbol: str, granularity: str, limit: int = 200) -> list[Candle]: ...

    # Orders
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        client_id: str | None = None,
        reduce_only: bool | None = None,
        leverage: int | None = None,
    ) -> Order: ...

    def cancel_order(self, order_id: str) -> bool: ...
    def get_order(self, order_id: str) -> Order | None: ...
    def list_orders(
        self, status: OrderStatus | None = None, symbol: str | None = None
    ) -> list[Order]: ...

    # Positions and fills
    def list_positions(self) -> list[Position]: ...
    def list_fills(self, symbol: str | None = None, limit: int = 200) -> list[dict]: ...

    # Streaming
    def stream_trades(self, symbols: Sequence[str]) -> Iterable[dict]: ...
    def stream_orderbook(self, symbols: Sequence[str], level: int = 1) -> Iterable[dict]: ...

    # Streaming observability (optional)
    def set_streaming_metrics_emitter(
        self, emitter: Callable[[dict[str, Any]], None] | None
    ) -> None: ...

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
from typing import Literal, Protocol


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
    qty: Decimal
    price: Decimal | None
    stop_price: Decimal | None
    tif: TimeInForce
    status: OrderStatus
    filled_qty: Decimal
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
        quantity: Decimal | None = None,
        qty: Decimal | None = None,
        filled_quantity: Decimal | None = None,
        filled_qty: Decimal | None = None,
    ) -> None:
        base_quantity = quantity if quantity is not None else qty
        if base_quantity is None:
            raise ValueError("Order requires a quantity")
        base_filled = filled_quantity if filled_quantity is not None else filled_qty
        if base_filled is None:
            base_filled = Decimal("0")

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
        self.qty = Decimal(str(base_quantity))
        self.filled_qty = Decimal(str(base_filled))

    @property
    def quantity(self) -> Decimal:
        """Preferred accessor for order size."""

        return self.qty

    @quantity.setter
    def quantity(self, value: Decimal) -> None:
        self.qty = Decimal(str(value))

    @property
    def filled_quantity(self) -> Decimal:
        """Preferred accessor for filled size."""

        return self.filled_qty

    @filled_quantity.setter
    def filled_quantity(self, value: Decimal) -> None:
        self.filled_qty = Decimal(str(value))


@dataclass
class Position:
    symbol: str
    qty: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    leverage: int | None
    side: Literal["long", "short"]

    @property
    def quantity(self) -> Decimal:
        """Preferred accessor for position size."""

        return self.qty

    @quantity.setter
    def quantity(self, value: Decimal) -> None:
        self.qty = Decimal(str(value))


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


class IBrokerage(Protocol):
    """Brokerage protocol supporting spot and derivatives."""

    # Connectivity
    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def validate_connection(self) -> bool: ...

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
        qty: Decimal,
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

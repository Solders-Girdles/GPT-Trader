"""
Brokerage core interfaces and shared types for spot and derivatives.

These interfaces define a stable contract for broker adapters (e.g., Coinbase),
allowing orchestration and strategies to remain broker-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, Iterable, List, Literal, Optional, Protocol, Sequence, Tuple


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
    min_notional: Optional[Decimal]
    price_increment: Decimal
    leverage_max: Optional[int] = None
    expiry: Optional[datetime] = None
    # Perpetuals-specific fields
    contract_size: Optional[Decimal] = None  # Size of one contract in base asset
    funding_rate: Optional[Decimal] = None  # Current funding rate (if available)
    next_funding_time: Optional[datetime] = None  # Next funding timestamp


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


@dataclass
class Order:
    id: str
    client_id: Optional[str]
    symbol: str
    side: OrderSide
    type: OrderType
    qty: Decimal
    price: Optional[Decimal]
    stop_price: Optional[Decimal]
    tif: TimeInForce
    status: OrderStatus
    filled_qty: Decimal
    avg_fill_price: Optional[Decimal]
    submitted_at: datetime
    updated_at: datetime


@dataclass
class Position:
    symbol: str
    qty: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    leverage: Optional[int]
    side: Literal["long", "short"]


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
    def list_balances(self) -> List[Balance]: ...

    # Products and market data
    def list_products(self, market: Optional[MarketType] = None) -> List[Product]: ...
    def get_product(self, symbol: str) -> Product: ...
    def get_quote(self, symbol: str) -> Quote: ...
    def get_candles(self, symbol: str, granularity: str, limit: int = 200) -> List[Candle]: ...

    # Orders
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        tif: TimeInForce = TimeInForce.GTC,
        client_id: Optional[str] = None,
        reduce_only: Optional[bool] = None,
        leverage: Optional[int] = None,
    ) -> Order: ...

    def cancel_order(self, order_id: str) -> bool: ...
    def get_order(self, order_id: str) -> Optional[Order]: ...
    def list_orders(self, status: Optional[OrderStatus] = None, symbol: Optional[str] = None) -> List[Order]: ...

    # Positions and fills
    def list_positions(self) -> List[Position]: ...
    def list_fills(self, symbol: Optional[str] = None, limit: int = 200) -> List[Dict]: ...

    # Streaming
    def stream_trades(self, symbols: Sequence[str]) -> Iterable[Dict]: ...
    def stream_orderbook(self, symbols: Sequence[str], level: int = 1) -> Iterable[Dict]: ...

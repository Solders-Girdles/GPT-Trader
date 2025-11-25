from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Protocol


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


class MarketType(str, Enum):
    SPOT = "SPOT"
    FUTURE = "FUTURE"
    PERPETUAL = "PERPETUAL"
    FUTURES = "FUTURES"  # Alias for compatibility


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


@dataclass
class Candle:
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass
class Quote:
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    ts: datetime


@dataclass
class Product:
    symbol: str
    base_asset: str
    quote_asset: str
    market_type: MarketType
    min_size: Decimal
    step_size: Decimal
    min_notional: Decimal | None
    price_increment: Decimal
    leverage_max: int | None
    expiry: datetime | None = None
    contract_size: Decimal | None = None
    funding_rate: Decimal | None = None
    next_funding_time: datetime | None = None


@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Decimal
    status: OrderStatus
    filled_quantity: Decimal = Decimal("0")
    price: Decimal | None = None
    stop_price: Decimal | None = None
    tif: TimeInForce = TimeInForce.GTC
    client_id: str | None = None
    avg_fill_price: Decimal | None = None
    submitted_at: datetime | None = None
    updated_at: datetime | None = None
    created_at: datetime | None = None


@dataclass
class Position:
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    side: str  # "long" or "short"
    leverage: int | None = None


@dataclass
class Balance:
    asset: str
    total: Decimal
    available: Decimal
    hold: Decimal = Decimal("0")


class IBrokerage(Protocol):
    def get_balances(self) -> list[Balance]: ...
    def get_positions(self) -> list[Position]: ...
    def list_balances(self) -> list[Balance]: ...
    def list_positions(self) -> list[Position]: ...
    def get_product(self, symbol: str) -> Product | None: ...
    def place_order(
        self,
        symbol: str | None = None,
        side: OrderSide | None = None,
        order_type: OrderType | None = None,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce | None = None,
        reduce_only: bool = False,
        leverage: int | None = None,
        client_id: str | None = None,
        order: Order | None = None,
    ) -> Order: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_order(self, order_id: str) -> Order: ...


class InvalidRequestError(Exception):
    pass


class InsufficientFunds(Exception):
    pass


class NotFoundError(Exception):
    pass


class AuthError(Exception):
    pass


class BrokerageError(Exception):
    pass


class RateLimitError(Exception):
    pass


class PermissionDeniedError(Exception):
    pass

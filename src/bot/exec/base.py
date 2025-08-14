from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Position:
    symbol: str
    qty: int
    avg_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float  # percentage
    current_price: float
    timestamp: datetime


@dataclass
class Order:
    id: str
    symbol: str
    side: str  # "buy" or "sell"
    qty: int
    filled_qty: int
    filled_avg_price: float
    status: str  # "new", "partially_filled", "filled", "canceled", "rejected"
    order_type: str  # "market", "limit", "stop", "stop_limit"
    time_in_force: str  # "day", "gtc", "ioc", "fok"
    created_at: datetime
    filled_at: datetime | None = None
    canceled_at: datetime | None = None
    limit_price: float | None = None
    stop_price: float | None = None


@dataclass
class Account:
    id: str
    account_number: str
    status: str
    crypto_status: str
    currency: str
    buying_power: float
    regt_buying_power: float
    daytrading_buying_power: float
    non_marginable_buying_power: float
    cash: float
    accrued_fees: float
    pending_transfer_out: float
    pending_transfer_in: float
    portfolio_value: float
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    created_at: datetime
    trade_suspended_by_user: bool
    multiplier: str
    shorting_enabled: bool
    equity: float
    last_equity: float
    long_market_value: float
    short_market_value: float
    initial_margin: float
    maintenance_margin: float
    last_maintenance_margin: float
    sma: float
    daytrade_count: int


class Broker(ABC):
    """Abstract base class for broker implementations."""

    @abstractmethod
    def get_account(self) -> Account:
        """Get current account information."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get current positions."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol."""
        ...

    @abstractmethod
    def submit_market_order(self, symbol: str, side: str, qty: int) -> Order:
        """Submit a market order."""
        ...

    @abstractmethod
    def submit_limit_order(self, symbol: str, side: str, qty: int, limit_price: float) -> Order:
        """Submit a limit order."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> Order | None:
        """Get order details."""
        ...

    @abstractmethod
    def get_orders(self, status: str | None = None, limit: int = 500) -> list[Order]:
        """Get orders with optional status filter."""
        ...

    @abstractmethod
    def get_bars(self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day") -> Any:
        """Get historical bars for a symbol."""
        ...

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> dict[str, Any] | None:
        """Get the latest bar for a symbol."""
        ...

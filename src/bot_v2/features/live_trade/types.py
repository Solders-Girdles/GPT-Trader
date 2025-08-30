"""
Local types for live trading.

Complete isolation - no external dependencies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Literal
from enum import Enum


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class BrokerConnection:
    """Broker connection information."""
    broker_name: str
    api_key: str
    api_secret: str
    is_paper: bool
    is_connected: bool
    account_id: Optional[str]
    base_url: Optional[str]


@dataclass
class Order:
    """Order information."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float]  # For limit orders
    stop_price: Optional[float]  # For stop orders
    status: OrderStatus
    submitted_at: datetime
    filled_at: Optional[datetime]
    filled_qty: int
    avg_fill_price: Optional[float]
    commission: float
    
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]


@dataclass
class Position:
    """Current position information."""
    symbol: str
    quantity: int
    side: Literal['long', 'short']
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    
    def get_value(self) -> float:
        """Get current position value."""
        return self.market_value


@dataclass
class AccountInfo:
    """Account information."""
    account_id: str
    cash: float
    portfolio_value: float
    buying_power: float
    positions_value: float
    margin_used: float
    pattern_day_trader: bool
    day_trades_remaining: int
    equity: float
    last_equity: float
    
    def get_available_cash(self) -> float:
        """Get available cash for trading."""
        return min(self.cash, self.buying_power)


@dataclass
class MarketHours:
    """Market hours information."""
    is_open: bool
    open_time: Optional[datetime]
    close_time: Optional[datetime]
    extended_hours_open: bool


@dataclass
class Quote:
    """Real-time quote data."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime
    
    def get_mid(self) -> float:
        """Get mid price."""
        return (self.bid + self.ask) / 2


@dataclass
class Bar:
    """OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class ExecutionReport:
    """Trade execution report."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    execution_id: str
"""
Local types for live trading.

Re-exports core types with deprecation warning.
Only defines types not present in core.
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Literal
from enum import Enum

# Re-export core types with deprecation warning
from ..brokerages.core.interfaces import (
    Order,
    Position,
    Quote,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce
)

# Emit deprecation warning when this module is imported
warnings.warn(
    "live_trade.types is deprecated for core types; use brokerages.core.interfaces directly",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    # Re-exported from core
    'Order', 'Position', 'Quote', 'OrderType', 'OrderSide', 'OrderStatus', 'TimeInForce',
    # Local-only types
    'BrokerConnection', 'AccountInfo', 'MarketHours', 'Bar', 'ExecutionReport'
]


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


# Local-only types (not in core interfaces)


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
"""Core account types used across all slices."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from gpt_trader.core.trading import OrderSide, OrderStatus, OrderType, TimeInForce


@dataclass
class Order:
    """Trading order representation."""

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
    """Trading position representation."""

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
    """Account balance for a single asset."""

    asset: str
    total: Decimal
    available: Decimal
    hold: Decimal = Decimal("0")

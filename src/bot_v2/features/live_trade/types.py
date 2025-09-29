"""Local types for live trading that complement the shared broker interfaces."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import cast

# Re-export core types with deprecation warning
from ..brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    TimeInForce,
)
from bot_v2.types.trading import AccountSnapshot, TradeFill, TradingPosition

__all__ = [
    # Re-exported from core
    "Order",
    "Position",
    "Quote",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "TimeInForce",
    # Local-only types
    "BrokerConnection",
    "AccountInfo",
    "MarketHours",
    "Bar",
    "ExecutionReport",
    "AccountSnapshot",
    "TradeFill",
    "TradingPosition",
    "position_to_trading_position",
]


def _to_decimal(value: float | int | Decimal) -> Decimal:
    """Convert numeric values to Decimal while avoiding float drift."""

    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass
class BrokerConnection:
    """Broker connection information."""

    broker_name: str
    api_key: str
    api_secret: str
    is_paper: bool
    is_connected: bool
    account_id: str | None
    base_url: str | None


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

    def to_account_snapshot(self) -> AccountSnapshot:
        """Convert account info to shared snapshot."""

        return AccountSnapshot(
            account_id=self.account_id,
            cash=_to_decimal(self.cash),
            equity=_to_decimal(self.equity),
            buying_power=_to_decimal(self.buying_power),
            positions_value=_to_decimal(self.positions_value),
            margin_used=_to_decimal(self.margin_used),
            pattern_day_trader=self.pattern_day_trader,
            day_trades_remaining=self.day_trades_remaining,
        )

    @classmethod
    def from_account_snapshot(cls, snapshot: AccountSnapshot) -> "AccountInfo":
        """Rehydrate AccountInfo from shared snapshot."""

        return cls(
            account_id=snapshot.account_id or "unknown",
            cash=float(snapshot.cash),
            portfolio_value=float(snapshot.equity),
            buying_power=float(snapshot.buying_power),
            positions_value=float(snapshot.positions_value),
            margin_used=float(snapshot.margin_used),
            pattern_day_trader=(
                bool(snapshot.pattern_day_trader)
                if snapshot.pattern_day_trader is not None
                else False
            ),
            day_trades_remaining=snapshot.day_trades_remaining or 0,
            equity=float(snapshot.equity),
            last_equity=float(snapshot.equity),
        )


@dataclass
class MarketHours:
    """Market hours information."""

    is_open: bool
    open_time: datetime | None
    close_time: datetime | None
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

    def to_trade_fill(self) -> TradeFill:
        """Convert execution report to shared trade fill."""

        return TradeFill(
            symbol=self.symbol,
            side=self.side,
            quantity=_to_decimal(self.quantity),
            price=_to_decimal(self.price),
            timestamp=self.timestamp,
            commission=_to_decimal(self.commission),
            slippage=Decimal("0"),
            order_id=self.order_id,
            execution_id=self.execution_id,
        )

    @classmethod
    def from_trade_fill(cls, fill: TradeFill) -> "ExecutionReport":
        """Create execution report from shared trade fill."""

        side = (
            cast(OrderSide, fill.side) if isinstance(fill.side, OrderSide) else OrderSide(fill.side)
        )
        return cls(
            order_id=fill.order_id or "",
            symbol=fill.symbol,
            side=side,
            quantity=int(fill.quantity),
            price=float(fill.price),
            commission=float(fill.commission),
            timestamp=fill.timestamp,
            execution_id=fill.execution_id or "",
        )


def position_to_trading_position(position: Position) -> TradingPosition:
    """Convert brokerage Position into shared trading position."""

    return TradingPosition(
        symbol=position.symbol,
        quantity=_to_decimal(position.quantity),
        entry_price=_to_decimal(position.entry_price),
        entry_timestamp=None,
        current_price=_to_decimal(position.mark_price),
        unrealized_pnl=_to_decimal(position.unrealized_pnl),
        realized_pnl=_to_decimal(position.realized_pnl),
        value=_to_decimal(position.quantity * position.mark_price),
    )

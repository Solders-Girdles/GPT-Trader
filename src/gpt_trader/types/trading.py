"""Shared trading domain types across execution modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal

from gpt_trader.features.brokerages.core.interfaces import OrderSide, OrderType


@dataclass
class TradingPosition:
    """Normalised position representation using Decimal precision."""

    symbol: str
    quantity: Decimal
    entry_price: Decimal
    entry_timestamp: datetime | None = None
    current_price: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    realized_pnl: Decimal | None = None
    value: Decimal | None = None


@dataclass
class AccountSnapshot:
    """Current account state shared across live, paper, and backtest contexts."""

    account_id: str | None
    cash: Decimal
    equity: Decimal
    buying_power: Decimal
    positions_value: Decimal
    margin_used: Decimal
    pattern_day_trader: bool | None = None
    day_trades_remaining: int | None = None


@dataclass
class TradeFill:
    """Executed trade event."""

    symbol: str
    side: OrderSide | Literal["buy", "sell"]
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    commission: Decimal = Decimal("0")
    slippage: Decimal | None = None
    order_id: str | None = None
    execution_id: str | None = None


@dataclass
class PerformanceSummary:
    """Common performance metrics surfaced by backtest and paper trading."""

    total_return: float
    max_drawdown: float
    sharpe_ratio: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    trades_count: int | None = None
    daily_return: float | None = None


@dataclass
class TradingSessionResult:
    """Aggregate outcome for a trading session or simulation."""

    start_time: datetime
    end_time: datetime | None
    account: AccountSnapshot
    positions: list[TradingPosition] = field(default_factory=list)
    fills: list[TradeFill] = field(default_factory=list)
    performance: PerformanceSummary | None = None


@dataclass
class OrderTicket:
    """Order request shared across execution layers."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: Literal["gtc", "ioc", "fok"] | None = None
    client_id: str | None = None


__all__ = [
    "TradingPosition",
    "AccountSnapshot",
    "TradeFill",
    "PerformanceSummary",
    "TradingSessionResult",
    "OrderTicket",
]

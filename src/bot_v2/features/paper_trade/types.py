"""
Local types for paper trading - no external dependencies.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, cast

import pandas as pd

from bot_v2.features.brokerages.core.interfaces import OrderSide
from bot_v2.types.trading import (
    AccountSnapshot,
    PerformanceSummary,
    TradeFill,
    TradingPosition,
    TradingSessionResult,
)


def _to_decimal(value: float | int | Decimal) -> Decimal:
    """Convert incoming numeric to Decimal without floating drift."""

    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass
class Position:
    """Current position in paper trading."""

    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    current_price: float
    unrealized_pnl: float
    value: float

    def to_trading_position(self) -> TradingPosition:
        """Convert to shared TradingPosition (Decimal based)."""

        return TradingPosition(
            symbol=self.symbol,
            quantity=_to_decimal(self.quantity),
            entry_price=_to_decimal(self.entry_price),
            entry_timestamp=self.entry_date,
            current_price=_to_decimal(self.current_price),
            unrealized_pnl=_to_decimal(self.unrealized_pnl),
            value=_to_decimal(self.value),
        )

    @classmethod
    def from_trading_position(cls, position: TradingPosition) -> "Position":
        """Create a paper-trade Position from shared model."""

        return cls(
            symbol=position.symbol,
            quantity=int(position.quantity),
            entry_price=float(position.entry_price),
            entry_date=position.entry_timestamp or datetime.utcnow(),
            current_price=float(position.current_price or position.entry_price),
            unrealized_pnl=float(position.unrealized_pnl or Decimal("0")),
            value=float(position.value or Decimal("0")),
        )


@dataclass
class TradeLog:
    """Record of a completed trade."""

    id: int
    symbol: str
    side: Literal["buy", "sell"]
    quantity: int
    price: float
    timestamp: datetime
    commission: float
    slippage: float

    def to_trade_fill(self) -> TradeFill:
        """Convert to shared TradeFill."""

        side_enum = OrderSide.BUY if self.side == "buy" else OrderSide.SELL
        return TradeFill(
            symbol=self.symbol,
            side=side_enum,
            quantity=_to_decimal(self.quantity),
            price=_to_decimal(self.price),
            timestamp=self.timestamp,
            commission=_to_decimal(self.commission),
            slippage=_to_decimal(self.slippage),
            order_id=str(self.id),
        )

    @classmethod
    def from_trade_fill(cls, fill: TradeFill, trade_id: int) -> "TradeLog":
        """Recreate TradeLog from shared model (best-effort)."""

        raw_side = fill.side.value if isinstance(fill.side, OrderSide) else str(fill.side)
        side = cast(Literal["buy", "sell"], raw_side.lower())
        return cls(
            id=trade_id,
            symbol=fill.symbol,
            side=side,
            quantity=int(fill.quantity),
            price=float(fill.price),
            timestamp=fill.timestamp,
            commission=float(fill.commission),
            slippage=float(fill.slippage or Decimal("0")),
        )


@dataclass
class AccountStatus:
    """Current account status."""

    cash: float
    positions_value: float
    total_equity: float
    buying_power: float
    margin_used: float
    day_trades_remaining: int

    def to_account_snapshot(self, account_id: str | None = None) -> AccountSnapshot:
        """Convert local account status to shared snapshot."""

        return AccountSnapshot(
            account_id=account_id,
            cash=_to_decimal(self.cash),
            equity=_to_decimal(self.total_equity),
            buying_power=_to_decimal(self.buying_power),
            positions_value=_to_decimal(self.positions_value),
            margin_used=_to_decimal(self.margin_used),
            pattern_day_trader=None,
            day_trades_remaining=self.day_trades_remaining,
        )

    @classmethod
    def from_account_snapshot(cls, snapshot: AccountSnapshot) -> "AccountStatus":
        """Build AccountStatus from shared snapshot."""

        return cls(
            cash=float(snapshot.cash),
            positions_value=float(snapshot.positions_value),
            total_equity=float(snapshot.equity),
            buying_power=float(snapshot.buying_power),
            margin_used=float(snapshot.margin_used),
            day_trades_remaining=snapshot.day_trades_remaining or 0,
        )


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""

    total_return: float
    daily_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades_count: int

    def to_performance_summary(self) -> PerformanceSummary:
        """Convert local metrics to shared PerformanceSummary."""

        return PerformanceSummary(
            total_return=self.total_return,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=self.sharpe_ratio,
            win_rate=self.win_rate,
            profit_factor=self.profit_factor,
            trades_count=self.trades_count,
            daily_return=self.daily_return,
        )


@dataclass
class PaperTradeResult:
    """Complete paper trading session results."""

    start_time: datetime
    end_time: datetime | None
    account_status: AccountStatus
    positions: list[Position]
    trade_log: list[TradeLog]
    performance: PerformanceMetrics
    equity_curve: pd.Series

    def summary(self) -> str:
        """Generate summary report."""
        duration = self.end_time - self.start_time if self.end_time else "Ongoing"

        return f"""
Paper Trading Summary
====================
Duration: {duration}
Total Equity: ${self.account_status.total_equity:,.2f}
Total Return: {self.performance.total_return:.2%}
Sharpe Ratio: {self.performance.sharpe_ratio:.2f}
Max Drawdown: {self.performance.max_drawdown:.2%}
Win Rate: {self.performance.win_rate:.2%}
Total Trades: {self.performance.trades_count}
Open Positions: {len(self.positions)}
        """.strip()

    def to_trading_session(self, account_id: str | None = None) -> TradingSessionResult:
        """Produce shared TradingSessionResult for downstream consumers."""

        return TradingSessionResult(
            start_time=self.start_time,
            end_time=self.end_time,
            account=self.account_status.to_account_snapshot(account_id=account_id),
            positions=[position.to_trading_position() for position in self.positions],
            fills=[trade.to_trade_fill() for trade in self.trade_log],
            performance=self.performance.to_performance_summary(),
        )

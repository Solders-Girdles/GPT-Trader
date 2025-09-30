"""Backtest-specific types and data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd
from bot_v2.features.brokerages.core.interfaces import OrderSide
from bot_v2.types.trading import (
    AccountSnapshot,
    PerformanceSummary,
    TradeFill,
    TradingSessionResult,
)

if TYPE_CHECKING:
    from pandas import Series
else:  # pragma: no cover - pandas runtime import already performed above
    Series = pd.Series


TradeDict = dict[str, Any]


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    trades: list[TradeDict] = field(default_factory=list)
    equity_curve: Series = field(default_factory=pd.Series)
    returns: Series = field(default_factory=pd.Series)
    metrics: BacktestMetrics | None = None
    initial_capital: float | Decimal | None = None

    def summary(self) -> str:
        """Get a summary of results."""
        if self.metrics:
            return (
                f"Total Return: {self.metrics.total_return:.2f}%\n"
                f"Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}\n"
                f"Max Drawdown: {self.metrics.max_drawdown:.2f}%\n"
                f"Win Rate: {self.metrics.win_rate:.2f}%\n"
                f"Total Trades: {self.metrics.total_trades}"
            )
        return "No results available"

    def final_equity(self) -> float:
        """Return the final equity value from the curve."""

        if len(self.equity_curve) == 0:
            return float(self.initial_capital or 0.0)
        return float(self.equity_curve.iloc[-1])

    def to_performance_summary(self) -> PerformanceSummary | None:
        """Convert metrics to shared performance summary."""

        if not self.metrics:
            return None
        return self.metrics.to_performance_summary()

    def to_trading_session(
        self,
        *,
        symbol: str | None = None,
        account_id: str | None = None,
        initial_capital: float | Decimal | None = None,
    ) -> TradingSessionResult:
        """Produce a shared trading session result for downstream consumers."""

        final_equity = Decimal(str(self.final_equity()))
        performance = self.to_performance_summary()

        account = AccountSnapshot(
            account_id=account_id,
            cash=final_equity,
            equity=final_equity,
            buying_power=final_equity,
            positions_value=Decimal("0"),
            margin_used=Decimal("0"),
            pattern_day_trader=None,
            day_trades_remaining=None,
        )

        fills = [_trade_dict_to_fill(trade, symbol=symbol) for trade in self.trades]

        return TradingSessionResult(
            start_time=(
                self.equity_curve.index[0].to_pydatetime()
                if len(self.equity_curve.index) > 0
                and isinstance(self.equity_curve.index[0], pd.Timestamp)
                else datetime.utcnow()
            ),
            end_time=(
                self.equity_curve.index[-1].to_pydatetime()
                if len(self.equity_curve.index) > 0
                and isinstance(self.equity_curve.index[-1], pd.Timestamp)
                else datetime.utcnow()
            ),
            account=account,
            positions=[],
            fills=fills,
            performance=performance,
        )


@dataclass
class BacktestMetrics:
    """Performance metrics for backtest."""

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    def to_performance_summary(self) -> PerformanceSummary:
        """Convert to the shared performance summary object."""

        return PerformanceSummary(
            total_return=self.total_return,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=self.sharpe_ratio,
            win_rate=self.win_rate,
            profit_factor=self.profit_factor,
            trades_count=self.total_trades,
            daily_return=None,
        )


def _trade_dict_to_fill(trade: TradeDict, *, symbol: str | None = None) -> TradeFill:
    """Convert legacy backtest trade dict into shared TradeFill."""

    raw_side = str(trade.get("side", "buy")).lower()
    side_enum = OrderSide.BUY if raw_side == "buy" else OrderSide.SELL
    quantity = Decimal(str(trade.get("quantity", 0)))
    price = Decimal(str(trade.get("price", 0)))
    commission_value = trade.get("commission")
    if commission_value is None:
        commission_value = trade.get("fee", 0)
    commission = Decimal(str(commission_value or 0))
    slippage_value = trade.get("slippage")
    if slippage_value is None:
        slippage_value = trade.get("slippage_amount", 0)
    slippage = Decimal(str(slippage_value or 0))
    trade_symbol = symbol or str(trade.get("symbol", ""))
    timestamp = trade.get("date") or trade.get("timestamp")
    if isinstance(timestamp, pd.Timestamp):
        ts = timestamp.to_pydatetime()
    elif isinstance(timestamp, datetime):
        ts = timestamp
    else:
        ts = datetime.utcnow()

    return TradeFill(
        symbol=trade_symbol,
        side=side_enum,
        quantity=quantity,
        price=price,
        timestamp=ts,
        commission=commission,
        slippage=slippage,
        order_id=str(trade.get("id", "")) or None,
        execution_id=None,
    )

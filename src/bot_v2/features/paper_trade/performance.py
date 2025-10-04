"""Performance tracking and result building for paper trading sessions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from bot_v2.features.paper_trade.types import PaperTradeResult, PerformanceMetrics
from bot_v2.types.trading import TradingSessionResult


@dataclass
class EquitySnapshot:
    """Point-in-time snapshot of account equity."""

    timestamp: datetime
    equity: float


class PerformanceTracker:
    """Tracks equity history and calculates performance metrics."""

    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = initial_capital
        self._history: list[EquitySnapshot] = []
        # Legacy compatibility store (list of dicts)
        self.legacy_history: list[dict[str, float | datetime]] = []

    def record(self, timestamp: datetime, equity: float) -> None:
        snapshot = EquitySnapshot(timestamp=timestamp, equity=float(equity))
        self._history.append(snapshot)
        self.legacy_history.append({"timestamp": snapshot.timestamp, "equity": snapshot.equity})

    def snapshots(self) -> list[EquitySnapshot]:
        """Return a copy of the recorded equity snapshots."""
        return list(self._history)

    def equity_series(self) -> pd.Series:
        """Build Pandas Series for equity curve (legacy behaviour)."""
        if not self._history:
            return pd.Series([self.initial_capital])

        timestamps = [snap.timestamp for snap in self._history]
        equity_values = [snap.equity for snap in self._history]
        return pd.Series(equity_values, index=timestamps)

    def replace_history(self, entries: Iterable[dict[str, Any]]) -> None:
        """Replace recorded history (legacy setter compatibility)."""
        self._history.clear()
        self.legacy_history.clear()
        for entry in entries:
            timestamp = entry.get("timestamp")
            equity = entry.get("equity")
            if timestamp is None or equity is None:
                continue
            if not isinstance(timestamp, datetime):
                try:
                    timestamp = datetime.fromisoformat(str(timestamp))
                except ValueError:
                    continue
            self.record(timestamp, float(equity))


class PerformanceCalculator:
    """Calculates performance metrics for paper trading sessions."""

    def __init__(self, tracker: PerformanceTracker) -> None:
        self.tracker = tracker

    def calculate(self, trade_log: Iterable, account_status: Any) -> PerformanceMetrics:
        """Compute performance metrics using equity history and trade log."""
        # Total return
        if self.tracker.legacy_history:
            final_equity = self.tracker.legacy_history[-1]["equity"]  # type: ignore[index]
            total_return = (
                final_equity - self.tracker.initial_capital
            ) / self.tracker.initial_capital
        else:
            total_return = (
                account_status.total_equity - self.tracker.initial_capital
            ) / self.tracker.initial_capital

        # Daily return & Sharpe
        if len(self.tracker.legacy_history) > 1:
            daily_returns: list[float] = []
            history = self.tracker.legacy_history
            for idx in range(1, len(history)):
                prev = history[idx - 1]["equity"]  # type: ignore[index]
                curr = history[idx]["equity"]  # type: ignore[index]
                daily_returns.append((curr - prev) / prev)

            if daily_returns:
                avg_daily_return = sum(daily_returns) / len(daily_returns)
                std_series = pd.Series(daily_returns)
                std_value = std_series.std() if not std_series.empty else 0
                sharpe_ratio = (
                    (avg_daily_return * 252) / (std_value * (252**0.5))
                    if std_value and std_value > 0
                    else 0
                )
            else:
                avg_daily_return = 0
                sharpe_ratio = 0
        else:
            avg_daily_return = 0
            sharpe_ratio = 0

        # Max drawdown
        if self.tracker.legacy_history:
            equity_values = [entry["equity"] for entry in self.tracker.legacy_history]
            peak = equity_values[0]
            max_dd = 0.0
            for value in equity_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_dd:
                    max_dd = drawdown
        else:
            max_dd = 0.0

        # Win rate & profit factor
        trades = list(trade_log)
        if trades:
            buy_trades = {t.id: t for t in trades if getattr(t, "side", None) == "buy"}
            sell_trades = [t for t in trades if getattr(t, "side", None) == "sell"]

            wins = 0
            losses = 0

            for sell in sell_trades:
                buy_id = sell.id - 1
                buy = buy_trades.get(buy_id)
                if buy:
                    pnl = (sell.price - buy.price) * sell.quantity
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            profit_factor = wins / losses if losses > 0 else float("inf") if wins > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0

        return PerformanceMetrics(
            total_return=total_return,
            daily_return=avg_daily_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades_count=len(trades),
        )


class ResultBuilder:
    """Builds paper trading results and trading session summaries."""

    def __init__(self, tracker: PerformanceTracker, calculator: PerformanceCalculator) -> None:
        self.tracker = tracker
        self.calculator = calculator

    def build_paper_result(
        self,
        start_time: datetime,
        end_time: datetime | None,
        account_status: Any,  # type: ignore[name-defined]
        positions: list,
        trade_log: list,
    ) -> PaperTradeResult:
        metrics = self.calculator.calculate(trade_log, account_status)

        equity_series = self.tracker.equity_series()

        return PaperTradeResult(
            start_time=start_time,
            end_time=end_time,
            account_status=account_status,
            positions=positions,
            trade_log=trade_log,
            performance=metrics,
            equity_curve=equity_series,
        )

    def build_trading_session(
        self,
        start_time: datetime,
        end_time: datetime | None,
        account_status: Any,
        positions: list,
        trade_log: list,
    ) -> TradingSessionResult:
        paper_result = self.build_paper_result(
            start_time,
            end_time,
            account_status,
            positions,
            trade_log,
        )
        return paper_result.to_trading_session()

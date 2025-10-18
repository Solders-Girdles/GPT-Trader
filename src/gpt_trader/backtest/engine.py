"""Deterministic backtest runner built around domain-level strategies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Any

from gpt_trader.domain import Bar, Strategy


@dataclass(frozen=True, slots=True)
class Trade:
    """Snapshot of a completed trade."""

    entry: Bar
    exit: Bar
    return_pct: Decimal
    hold_duration: timedelta
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Aggregate result produced after running a backtest."""

    symbol: str
    trades: list[Trade]
    cumulative_return: Decimal
    average_trade_return: Decimal
    win_rate: Decimal
    total_trades: int
    best_trade_return: Decimal
    worst_trade_return: Decimal
    max_drawdown: Decimal
    equity_curve: list[Decimal]


class Backtester:
    """Single-strategy backtest runner."""

    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def run(self, symbol: str, bars: Sequence[Bar]) -> BacktestResult:
        if not bars:
            raise ValueError("Backtester requires at least one bar.")

        completed_trades: list[Trade] = []
        position_entry: Bar | None = None
        entry_snapshot: dict[str, Any] | None = None
        cumulative_return = Decimal("0")
        rolling_window: list[Bar] = []
        equity = Decimal("1")
        equity_curve: list[Decimal] = [equity]

        for bar in bars:
            rolling_window.append(bar)
            signal = self._strategy.decide(list(rolling_window))

            if signal.action == "BUY":
                if position_entry is None:
                    position_entry = bar
                    entry_snapshot = {
                        "metadata": signal.metadata,
                        "confidence": signal.confidence,
                    }
                continue

            if signal.action == "SELL" and position_entry is not None:
                entry_price = position_entry.close
                exit_price = bar.close
                return_pct = (exit_price - entry_price) / entry_price
                hold_duration = bar.timestamp - position_entry.timestamp
                exit_snapshot = {
                    "metadata": signal.metadata,
                    "confidence": signal.confidence,
                }
                combined_metadata = _compose_trade_metadata(entry_snapshot, exit_snapshot)
                trade = Trade(
                    entry=position_entry,
                    exit=bar,
                    return_pct=return_pct,
                    hold_duration=hold_duration,
                    metadata=combined_metadata,
                )
                completed_trades.append(trade)
                cumulative_return += return_pct
                equity *= Decimal("1") + return_pct
                equity_curve.append(equity)
                position_entry = None
                entry_snapshot = None

        total_trades = len(completed_trades)
        average_trade_return = (
            cumulative_return / Decimal(total_trades) if total_trades else Decimal("0")
        )
        wins = sum(1 for trade in completed_trades if trade.return_pct > 0)
        win_rate = Decimal(wins) / Decimal(total_trades) if total_trades else Decimal("0")

        if total_trades:
            best_trade_return = max(trade.return_pct for trade in completed_trades)
            worst_trade_return = min(trade.return_pct for trade in completed_trades)
        else:
            best_trade_return = Decimal("0")
            worst_trade_return = Decimal("0")

        peak = equity_curve[0]
        max_drawdown = Decimal("0")
        for level in equity_curve[1:]:
            if level > peak:
                peak = level
            else:
                drawdown = (peak - level) / peak if peak != 0 else Decimal("0")
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        return BacktestResult(
            symbol=symbol,
            trades=completed_trades,
            cumulative_return=cumulative_return,
            average_trade_return=average_trade_return,
            win_rate=win_rate,
            total_trades=total_trades,
            best_trade_return=best_trade_return,
            worst_trade_return=worst_trade_return,
            max_drawdown=max_drawdown,
            equity_curve=equity_curve,
        )


def _compose_trade_metadata(
    entry_snapshot: dict[str, Any] | None, exit_snapshot: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not entry_snapshot and not exit_snapshot:
        return None

    metadata: dict[str, Any] = {}
    if entry_snapshot:
        metadata["entry"] = _normalise_snapshot(entry_snapshot)
    if exit_snapshot:
        metadata["exit"] = _normalise_snapshot(exit_snapshot)
    return metadata


def _normalise_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    raw_meta = snapshot.get("metadata")
    confidence = snapshot.get("confidence")
    reason = _extract_reason(raw_meta)
    return {
        "reason": reason,
        "confidence": float(confidence) if isinstance(confidence, (float, int)) else None,
        "metadata": raw_meta,
    }


def _extract_reason(raw_meta: Any) -> str | None:
    if isinstance(raw_meta, dict):
        reason = raw_meta.get("reason")
        if isinstance(reason, str):
            return reason
    return None


__all__ = ["BacktestResult", "Backtester", "Trade"]

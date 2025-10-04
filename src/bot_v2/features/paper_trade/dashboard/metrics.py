"""Metrics aggregation helper for the paper trading dashboard."""

from __future__ import annotations

from typing import Any


class DashboardMetricsAssembler:
    """Computes dashboard metrics from the paper trading engine state."""

    def __init__(self, initial_equity: float) -> None:
        self.initial_equity = initial_equity

    def calculate(self, engine: Any) -> dict[str, Any]:
        equity = engine.calculate_equity()

        # Returns
        if abs(self.initial_equity) < 1e-9:
            returns_pct = 0.0
        else:
            returns_pct = round(((equity - self.initial_equity) / self.initial_equity) * 100, 2)

        # Drawdown (legacy behaviour – uses current equity as peak proxy)
        peak_equity = self.initial_equity
        for _ in engine.trades:
            peak_equity = max(peak_equity, equity)

        drawdown = 0.0
        if peak_equity > 0:
            drawdown = ((peak_equity - equity) / peak_equity) * 100

        # Win / loss stats
        winning_trades = sum(
            1 for trade in engine.trades if getattr(trade, "pnl", 0) and trade.pnl > 0
        )
        losing_trades = sum(
            1 for trade in engine.trades if getattr(trade, "pnl", 0) and trade.pnl < 0
        )
        total_closed = winning_trades + losing_trades
        win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0.0

        # Exposure
        positions_value = sum(
            position.quantity
            * (position.current_price if position.current_price > 0 else position.entry_price)
            for position in engine.positions.values()
        )
        exposure_pct = (positions_value / equity * 100) if equity > 0 else 0.0

        return {
            "equity": equity,
            "cash": engine.cash,
            "positions_value": positions_value,
            "returns_pct": returns_pct,
            "drawdown_pct": drawdown,
            "win_rate": win_rate,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_trades": len(engine.trades),
            "exposure_pct": exposure_pct,
            "positions_count": len(engine.positions),
        }

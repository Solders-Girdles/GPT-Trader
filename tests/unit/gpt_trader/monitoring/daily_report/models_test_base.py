"""Shared helpers for daily report model tests."""

from __future__ import annotations

from gpt_trader.monitoring.daily_report.models import DailyReport, SymbolPerformance


def _create_symbol_performance(
    symbol: str = "BTC-USD",
    regime: str | None = "trending",
    realized_pnl: float = 100.0,
    unrealized_pnl: float = 50.0,
    funding_pnl: float = -5.0,
    total_pnl: float = 145.0,
    trades: int = 10,
    win_rate: float = 0.6,
) -> SymbolPerformance:
    """Create a test SymbolPerformance instance."""
    return SymbolPerformance(
        symbol=symbol,
        regime=regime,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        funding_pnl=funding_pnl,
        total_pnl=total_pnl,
        trades=trades,
        win_rate=win_rate,
    )


def _create_daily_report(
    symbol_performance: list[SymbolPerformance] | None = None,
    guard_triggers: dict[str, int] | None = None,
    circuit_breaker_state: dict | None = None,
    equity: float | None = 100000.0,
    equity_change_pct: float | None = 0.5,
) -> DailyReport:
    """Create a test DailyReport instance."""
    return DailyReport(
        date="2024-01-15",
        profile="PROD",
        generated_at="2024-01-15T12:00:00",
        equity=equity,
        equity_change=500.0,
        equity_change_pct=equity_change_pct,
        realized_pnl=400.0,
        unrealized_pnl=100.0,
        funding_pnl=-25.0,
        total_pnl=475.0,
        fees_paid=50.0,
        win_rate=0.65,
        profit_factor=2.5,
        sharpe_ratio=1.8,
        max_drawdown=5000.0,
        max_drawdown_pct=5.0,
        total_trades=20,
        winning_trades=13,
        losing_trades=7,
        avg_win=100.0,
        avg_loss=-50.0,
        largest_win=500.0,
        largest_loss=-200.0,
        guard_triggers=guard_triggers or {},
        circuit_breaker_state=circuit_breaker_state or {},
        symbol_performance=symbol_performance or [],
        stale_marks_count=2,
        ws_reconnects=1,
        unfilled_orders=0,
        api_errors=3,
    )

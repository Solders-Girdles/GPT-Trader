"""Datamodels for daily performance reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SymbolPerformance:
    """Performance metrics for a single symbol."""

    symbol: str
    regime: str | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    funding_pnl: float = 0.0
    total_pnl: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    exposure_usd: float = 0.0


@dataclass
class DailyReport:
    """Complete daily performance report."""

    date: str
    profile: str
    generated_at: str
    equity: float | None
    equity_change: float
    equity_change_pct: float | None
    realized_pnl: float
    unrealized_pnl: float
    funding_pnl: float
    total_pnl: float
    fees_paid: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    guard_triggers: dict[str, int]
    circuit_breaker_state: dict[str, Any]
    symbol_performance: list[SymbolPerformance]
    stale_marks_count: int
    ws_reconnects: int
    unfilled_orders: int
    api_errors: int
    liveness: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        payload = {
            "date": self.date,
            "profile": self.profile,
            "generated_at": self.generated_at,
            "account": {
                "equity": self.equity,
                "equity_change": self.equity_change,
                "equity_change_pct": self.equity_change_pct,
            },
            "pnl": {
                "realized": self.realized_pnl,
                "unrealized": self.unrealized_pnl,
                "funding": self.funding_pnl,
                "total": self.total_pnl,
                "fees": self.fees_paid,
            },
            "performance": {
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "max_drawdown_pct": self.max_drawdown_pct,
            },
            "trades": {
                "total": self.total_trades,
                "winning": self.winning_trades,
                "losing": self.losing_trades,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "largest_win": self.largest_win,
                "largest_loss": self.largest_loss,
            },
            "risk": {
                "guard_triggers": self.guard_triggers,
                "circuit_breaker_state": self.circuit_breaker_state,
            },
            "symbols": [
                {
                    "symbol": sp.symbol,
                    "regime": sp.regime,
                    "pnl": {
                        "realized": sp.realized_pnl,
                        "unrealized": sp.unrealized_pnl,
                        "funding": sp.funding_pnl,
                        "total": sp.total_pnl,
                    },
                    "trades": sp.trades,
                    "win_rate": sp.win_rate,
                    "profit_factor": sp.profit_factor,
                    "exposure_usd": sp.exposure_usd,
                }
                for sp in self.symbol_performance
            ],
            "health": {
                "stale_marks": self.stale_marks_count,
                "ws_reconnects": self.ws_reconnects,
                "unfilled_orders": self.unfilled_orders,
                "api_errors": self.api_errors,
            },
        }
        if self.liveness is not None:
            payload["health"]["liveness"] = self.liveness
        if self.runtime is not None:
            payload["runtime"] = self.runtime
        return payload

    def to_text(self) -> str:
        """Format as human-readable text report."""
        equity_text = "N/A" if self.equity is None else f"${self.equity:,.2f}"
        change_pct_text = (
            "N/A" if self.equity_change_pct is None else f"{self.equity_change_pct:+.2f}%"
        )
        lines = [
            "=" * 80,
            f"Daily Trading Report - {self.date}",
            f"Profile: {self.profile}",
            f"Generated: {self.generated_at}",
            "=" * 80,
            "",
        ]

        if self.liveness or self.runtime:
            lines.extend(
                [
                    "RUNTIME",
                    "-" * 80,
                ]
            )
            if self.liveness:
                status = self.liveness.get("status", "UNKNOWN")
                events = self.liveness.get("events", {})
                heartbeat_age = events.get("heartbeat", {}).get("age_seconds")
                price_tick_age = events.get("price_tick", {}).get("age_seconds")
                lines.append(
                    "  Liveness:        "
                    f"{status} (heartbeat_age={heartbeat_age}, price_tick_age={price_tick_age})"
                )
            if self.runtime:
                build_sha = self.runtime.get("build_sha") or self.runtime.get("buildSha")
                if build_sha:
                    lines.append(f"  Build SHA:       {build_sha}")
            lines.append("")

        lines.extend(
            [
                "ACCOUNT SUMMARY",
                "-" * 80,
                f"  Equity:          {equity_text}",
                f"  Change (24h):    ${self.equity_change:+,.2f} ({change_pct_text})",
                "",
                "PnL BREAKDOWN",
                "-" * 80,
                f"  Realized PnL:    ${self.realized_pnl:+,.2f}",
                f"  Unrealized PnL:  ${self.unrealized_pnl:+,.2f}",
                f"  Funding PnL:     ${self.funding_pnl:+,.2f}",
                f"  Fees Paid:       ${self.fees_paid:,.2f}",
                f"  Total PnL:       ${self.total_pnl:+,.2f}",
                "",
                "PERFORMANCE METRICS",
                "-" * 80,
                f"  Win Rate:        {self.win_rate:.2%}",
                f"  Profit Factor:   {self.profit_factor:.2f}",
                f"  Sharpe Ratio:    {self.sharpe_ratio:.2f}",
                f"  Max Drawdown:    ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2f}%)",
                "",
                "TRADE STATISTICS",
                "-" * 80,
                f"  Total Trades:    {self.total_trades}",
                f"  Winning:         {self.winning_trades}",
                f"  Losing:          {self.losing_trades}",
                f"  Avg Win:         ${self.avg_win:,.2f}",
                f"  Avg Loss:        ${self.avg_loss:,.2f}",
                f"  Largest Win:     ${self.largest_win:,.2f}",
                f"  Largest Loss:    ${self.largest_loss:,.2f}",
                "",
            ]
        )

        if self.symbol_performance:
            lines.extend(
                [
                    "TOP PERFORMERS BY SYMBOL",
                    "-" * 80,
                    f"{'Symbol':<12} {'Regime':<10} {'Total PnL':>12} {'Realized':>12} {'Unreal':>12} {'Funding':>12} {'Trades':>8} {'Win%':>8}",
                    "-" * 80,
                ]
            )
            sorted_symbols = sorted(
                self.symbol_performance,
                key=lambda sp: sp.total_pnl,
                reverse=True,
            )
            for sp in sorted_symbols[:10]:
                regime = sp.regime or "N/A"
                lines.append(
                    f"{sp.symbol:<12} {regime:<10} "
                    f"${sp.total_pnl:>11,.2f} "
                    f"${sp.realized_pnl:>11,.2f} "
                    f"${sp.unrealized_pnl:>11,.2f} "
                    f"${sp.funding_pnl:>11,.2f} "
                    f"{sp.trades:>8} "
                    f"{sp.win_rate:>7.1%}"
                )
            lines.append("")

        lines.extend(
            [
                "RISK CONTROLS",
                "-" * 80,
            ]
        )
        if self.guard_triggers:
            lines.append("  Guard Triggers:")
            for guard, count in sorted(self.guard_triggers.items()):
                lines.append(f"    {guard}: {count}")
        else:
            lines.append("  No guard triggers")

        if self.circuit_breaker_state:
            lines.append("  Circuit Breaker State:")
            for key, value in self.circuit_breaker_state.items():
                lines.append(f"    {key}: {value}")
        else:
            lines.append("  Circuit breakers: OK")
        lines.append("")

        lines.extend(
            [
                "HEALTH METRICS",
                "-" * 80,
                f"  Stale Marks:     {self.stale_marks_count}",
                f"  WS Reconnects:   {self.ws_reconnects}",
                f"  Unfilled Orders: {self.unfilled_orders}",
                f"  API Errors:      {self.api_errors}",
                "",
                "=" * 80,
            ]
        )

        return "\n".join(lines)


__all__ = ["SymbolPerformance", "DailyReport"]

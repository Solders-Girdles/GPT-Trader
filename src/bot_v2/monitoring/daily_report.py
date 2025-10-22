"""
Daily performance report generator.

Generates comprehensive daily reports including:
- Equity, realized/unrealized PnL, funding PnL
- Guard triggers and circuit breaker state
- Win rate, Profit Factor, Sharpe ratio, Max drawdown
- Top drivers: symbol × regime PnL breakdown
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="daily_report")


@dataclass
class SymbolPerformance:
    """Performance metrics for a single symbol."""

    symbol: str
    regime: str | None = None  # Market regime during trading
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

    # Report metadata
    date: str
    profile: str
    generated_at: str

    # Account metrics
    equity: float
    equity_change: float
    equity_change_pct: float

    # PnL breakdown
    realized_pnl: float
    unrealized_pnl: float
    funding_pnl: float
    total_pnl: float
    fees_paid: float

    # Performance metrics
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float

    # Risk metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Guard activity
    guard_triggers: dict[str, int]
    circuit_breaker_state: dict[str, Any]

    # Symbol breakdown
    symbol_performance: list[SymbolPerformance]

    # Health metrics
    stale_marks_count: int
    ws_reconnects: int
    unfilled_orders: int
    api_errors: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
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

    def to_text(self) -> str:
        """Format as human-readable text report."""
        lines = [
            "=" * 80,
            f"Daily Trading Report - {self.date}",
            f"Profile: {self.profile}",
            f"Generated: {self.generated_at}",
            "=" * 80,
            "",
            "ACCOUNT SUMMARY",
            "-" * 80,
            f"  Equity:          ${self.equity:,.2f}",
            f"  Change (24h):    ${self.equity_change:+,.2f} ({self.equity_change_pct:+.2f}%)",
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

        # Symbol performance table
        if self.symbol_performance:
            lines.extend([
                "TOP PERFORMERS BY SYMBOL",
                "-" * 80,
                f"{'Symbol':<12} {'Regime':<10} {'Total PnL':>12} {'Realized':>12} {'Unreal':>12} {'Funding':>12} {'Trades':>8} {'Win%':>8}",
                "-" * 80,
            ])

            # Sort by total PnL
            sorted_symbols = sorted(self.symbol_performance, key=lambda x: x.total_pnl, reverse=True)
            for sp in sorted_symbols[:10]:  # Top 10
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

        # Risk controls
        lines.extend([
            "RISK CONTROLS",
            "-" * 80,
        ])
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

        # Health metrics
        lines.extend([
            "HEALTH METRICS",
            "-" * 80,
            f"  Stale Marks:     {self.stale_marks_count}",
            f"  WS Reconnects:   {self.ws_reconnects}",
            f"  Unfilled Orders: {self.unfilled_orders}",
            f"  API Errors:      {self.api_errors}",
            "",
            "=" * 80,
        ])

        return "\n".join(lines)


class DailyReportGenerator:
    """Generates daily performance reports from event store and metrics."""

    def __init__(self, profile: str = "demo", data_dir: Path | None = None):
        """
        Initialize report generator.

        Args:
            profile: Trading profile (demo, prod, etc.)
            data_dir: Base data directory (defaults to var/data/coinbase_trader)
        """
        self.profile = profile
        if data_dir is None:
            data_dir = Path("var/data/coinbase_trader") / profile
        self.data_dir = Path(data_dir)
        self.events_file = self.data_dir / "events.jsonl"
        self.metrics_file = self.data_dir / "metrics.json"

    def generate(
        self, date: datetime | None = None, lookback_hours: int = 24
    ) -> DailyReport:
        """
        Generate daily report.

        Args:
            date: Date to generate report for (defaults to today)
            lookback_hours: Hours to look back for data

        Returns:
            Daily report
        """
        if date is None:
            date = datetime.now()

        logger.info(f"Generating daily report for {date.date()} (profile={self.profile})")

        # Load current metrics
        current_metrics = self._load_metrics()

        # Parse events for the lookback period
        cutoff = date - timedelta(hours=lookback_hours)
        events = self._load_events_since(cutoff)

        # Calculate metrics
        pnl_metrics = self._calculate_pnl(events, current_metrics)
        trade_metrics = self._calculate_trade_metrics(events)
        symbol_metrics = self._calculate_symbol_metrics(events)
        risk_metrics = self._calculate_risk_metrics(events)
        health_metrics = self._calculate_health_metrics(events)

        # Build report
        return DailyReport(
            date=date.strftime("%Y-%m-%d"),
            profile=self.profile,
            generated_at=datetime.now().isoformat(),
            **pnl_metrics,
            **trade_metrics,
            symbol_performance=symbol_metrics,
            **risk_metrics,
            **health_metrics,
        )

    def _load_metrics(self) -> dict[str, Any]:
        """Load current metrics from metrics.json."""
        if not self.metrics_file.exists():
            logger.warning(f"Metrics file not found: {self.metrics_file}")
            return {}

        try:
            with open(self.metrics_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return {}

    def _load_events_since(self, cutoff: datetime) -> list[dict[str, Any]]:
        """Load events since cutoff time."""
        if not self.events_file.exists():
            logger.warning(f"Events file not found: {self.events_file}")
            return []

        events = []
        try:
            with open(self.events_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        # Parse timestamp
                        ts_str = event.get("timestamp", "")
                        if ts_str:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if ts >= cutoff:
                                events.append(event)
                    except Exception as e:
                        logger.debug(f"Failed to parse event: {e}")
                        continue
        except Exception as e:
            logger.error(f"Failed to load events: {e}")

        logger.info(f"Loaded {len(events)} events since {cutoff}")
        return events

    def _calculate_pnl(
        self, events: list[dict[str, Any]], current_metrics: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate PnL metrics."""
        # Get equity from current metrics
        equity = float(current_metrics.get("account", {}).get("equity", 0))

        # Calculate PnL from events
        realized = 0.0
        unrealized = 0.0
        funding = 0.0
        fees = 0.0

        for event in events:
            event_type = event.get("type", "")
            if event_type == "pnl_update":
                realized += float(event.get("realized_pnl", 0))
                unrealized = float(event.get("unrealized_pnl", 0))  # Latest snapshot
            elif event_type == "funding_payment":
                funding += float(event.get("amount", 0))
            elif event_type == "fill":
                fees += float(event.get("fee", 0))

        # Estimate previous equity
        total_pnl = realized + unrealized
        prev_equity = equity - total_pnl
        equity_change = total_pnl
        equity_change_pct = (equity_change / prev_equity * 100) if prev_equity > 0 else 0

        return {
            "equity": equity,
            "equity_change": equity_change,
            "equity_change_pct": equity_change_pct,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "funding_pnl": funding,
            "total_pnl": total_pnl,
            "fees_paid": fees,
        }

    def _calculate_trade_metrics(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate trade statistics and performance metrics."""
        fills = [e for e in events if e.get("type") == "fill"]

        wins = []
        losses = []

        # Group fills by order to calculate trade PnL
        # Simplified: just look at fill-level PnL if available
        for fill in fills:
            pnl = fill.get("pnl", 0)
            if pnl > 0:
                wins.append(pnl)
            elif pnl < 0:
                losses.append(abs(pnl))

        total_trades = len(wins) + len(losses)
        winning_trades = len(wins)
        losing_trades = len(losses)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = max(losses) if losses else 0

        # Profit factor: gross profit / gross loss
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio: simplified calculation
        # Would need returns series for proper calculation
        sharpe_ratio = 0.0
        if total_trades > 0:
            returns = [w for w in wins] + [-l for l in losses]
            if len(returns) > 1:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                std_dev = variance ** 0.5
                sharpe_ratio = (mean_return / std_dev) if std_dev > 0 else 0

        # Max drawdown
        equity_series = []
        running_equity = 0.0
        for event in sorted(events, key=lambda e: e.get("timestamp", "")):
            if event.get("type") == "fill":
                running_equity += event.get("pnl", 0)
                equity_series.append(running_equity)

        max_dd = 0.0
        max_dd_pct = 0.0
        if equity_series:
            peak = equity_series[0]
            for equity in equity_series:
                if equity > peak:
                    peak = equity
                drawdown = peak - equity
                if drawdown > max_dd:
                    max_dd = drawdown
                    max_dd_pct = (drawdown / peak * 100) if peak > 0 else 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd,
            "max_drawdown_pct": max_dd_pct,
        }

    def _calculate_symbol_metrics(
        self, events: list[dict[str, Any]]
    ) -> list[SymbolPerformance]:
        """Calculate per-symbol performance."""
        symbol_data: dict[str, dict[str, Any]] = {}

        for event in events:
            symbol = event.get("symbol")
            if not symbol:
                continue

            if symbol not in symbol_data:
                symbol_data[symbol] = {
                    "realized": 0.0,
                    "unrealized": 0.0,
                    "funding": 0.0,
                    "trades": 0,
                    "wins": [],
                    "losses": [],
                    "regime": None,
                    "exposure": 0.0,
                }

            event_type = event.get("type", "")
            if event_type == "fill":
                symbol_data[symbol]["trades"] += 1
                pnl = event.get("pnl", 0)
                symbol_data[symbol]["realized"] += pnl
                if pnl > 0:
                    symbol_data[symbol]["wins"].append(pnl)
                elif pnl < 0:
                    symbol_data[symbol]["losses"].append(abs(pnl))

            elif event_type == "pnl_update" and event.get("symbol") == symbol:
                symbol_data[symbol]["unrealized"] = event.get("unrealized_pnl", 0)

            elif event_type == "funding_payment" and event.get("symbol") == symbol:
                symbol_data[symbol]["funding"] += event.get("amount", 0)

            # Track regime if available
            if "regime" in event:
                symbol_data[symbol]["regime"] = event["regime"]

            # Track exposure
            if event_type == "position_update":
                qty = event.get("quantity", 0)
                price = event.get("price", 0)
                symbol_data[symbol]["exposure"] = abs(qty * price)

        # Convert to SymbolPerformance objects
        performances = []
        for symbol, data in symbol_data.items():
            wins = data["wins"]
            losses = data["losses"]
            total_trades = len(wins) + len(losses)

            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            gross_profit = sum(wins)
            gross_loss = sum(losses)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            performances.append(
                SymbolPerformance(
                    symbol=symbol,
                    regime=data["regime"],
                    realized_pnl=data["realized"],
                    unrealized_pnl=data["unrealized"],
                    funding_pnl=data["funding"],
                    total_pnl=data["realized"] + data["unrealized"] + data["funding"],
                    trades=data["trades"],
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    profit_factor=profit_factor,
                    exposure_usd=data["exposure"],
                )
            )

        return performances

    def _calculate_risk_metrics(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate risk control metrics."""
        guard_triggers: dict[str, int] = {}
        circuit_breaker_state = {}

        for event in events:
            event_type = event.get("type", "")

            if event_type == "guard_triggered":
                guard_name = event.get("guard", "unknown")
                guard_triggers[guard_name] = guard_triggers.get(guard_name, 0) + 1

            elif event_type == "circuit_breaker_triggered":
                # Track latest state
                circuit_breaker_state = {
                    "triggered": True,
                    "rule": event.get("rule"),
                    "action": event.get("action"),
                    "timestamp": event.get("timestamp"),
                }

        return {
            "guard_triggers": guard_triggers,
            "circuit_breaker_state": circuit_breaker_state,
        }

    def _calculate_health_metrics(self, events: list[dict[str, Any]]) -> dict[str, int]:
        """Calculate health metrics."""
        stale_marks = 0
        ws_reconnects = 0
        unfilled = 0
        api_errors = 0

        for event in events:
            event_type = event.get("type", "")

            if event_type == "stale_mark_detected":
                stale_marks += 1
            elif event_type == "websocket_reconnect":
                ws_reconnects += 1
            elif event_type == "unfilled_order_alert":
                unfilled += 1
            elif event_type == "api_error":
                api_errors += 1

        return {
            "stale_marks_count": stale_marks,
            "ws_reconnects": ws_reconnects,
            "unfilled_orders": unfilled,
            "api_errors": api_errors,
        }

    def save_report(self, report: DailyReport, output_dir: Path | None = None) -> Path:
        """
        Save report to file.

        Args:
            report: Report to save
            output_dir: Output directory (defaults to data_dir/reports)

        Returns:
            Path to saved report
        """
        if output_dir is None:
            output_dir = self.data_dir / "reports"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = output_dir / f"daily_report_{report.date}.json"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Saved JSON report to {json_path}")

        # Save as text
        text_path = output_dir / f"daily_report_{report.date}.txt"
        with open(text_path, "w") as f:
            f.write(report.to_text())
        logger.info(f"Saved text report to {text_path}")

        return text_path

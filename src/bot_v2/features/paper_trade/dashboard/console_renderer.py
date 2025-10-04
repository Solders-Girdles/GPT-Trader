"""Console rendering helpers for the paper trading dashboard."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from bot_v2.features.paper_trade.dashboard.formatters import DashboardFormatter

logger = logging.getLogger(__name__)


class ConsoleRenderer:
    """Renders dashboard sections to stdout."""

    def __init__(self, formatter: DashboardFormatter, start_time: datetime) -> None:
        self.formatter = formatter
        self.start_time = start_time

    def render_header(self, *, bot_id: str) -> None:
        header = [
            "=" * 80,
            "                        PAPER TRADING DASHBOARD",
            "=" * 80,
            f"Bot ID: {bot_id}",
            f"Runtime: {datetime.now() - self.start_time}",
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "-" * 80,
        ]
        for line in header:
            print(line)
        logger.info("Dashboard header displayed for bot %s", bot_id)

    def render_portfolio_summary(self, metrics: dict[str, Any]) -> None:
        summary_lines = [
            "\n📊 PORTFOLIO SUMMARY",
            "-" * 40,
            f"{'Equity:':<20} {self.formatter.currency_str(metrics['equity']):<20} "
            f"{'Returns:':<15} {self.formatter.percentage_str(metrics['returns_pct'])}",
            f"{'Cash:':<20} {self.formatter.currency_str(metrics['cash']):<20} "
            f"{'Drawdown:':<15} {self.formatter.percentage_str(metrics['drawdown_pct'])}",
            f"{'Positions Value:':<20} {self.formatter.currency_str(metrics['positions_value']):<20} "
            f"{'Exposure:':<15} {self.formatter.percentage_str(metrics['exposure_pct'])}",
        ]
        for line in summary_lines:
            print(line)
        logger.debug(
            "Portfolio summary: equity=%s, returns=%s%%", metrics["equity"], metrics["returns_pct"]
        )

    def render_positions(self, positions: dict[str, Any]) -> None:
        print("\n📈 OPEN POSITIONS")
        print("-" * 40)

        if not positions:
            print("No open positions")
            logger.debug("No open positions")
            return

        print(f"{'Symbol':<15} {'Qty':<12} {'Entry':<12} {'Current':<12} {'P&L':<12} {'P&L %'}")
        print("-" * 75)

        for symbol, pos in positions.items():
            current = pos.current_price if pos.current_price > 0 else pos.entry_price
            pnl = (current - pos.entry_price) * pos.quantity
            pnl_pct = ((current / pos.entry_price) - 1) * 100 if pos.entry_price > 0 else 0

            print(
                f"{symbol:<15} {pos.quantity:<12.6f} "
                f"${pos.entry_price:<11.2f} ${current:<11.2f} "
                f"${pnl:<11.2f} {self.formatter.percentage_str(pnl_pct)}"
            )
        logger.debug("Displayed %d open positions", len(positions))

    def render_performance(self, metrics: dict[str, Any]) -> None:
        performance_lines = [
            "\n📈 PERFORMANCE METRICS",
            "-" * 40,
            f"{'Total Trades:':<20} {metrics['total_trades']}",
            f"{'Winning Trades:':<20} {metrics['winning_trades']}",
            f"{'Losing Trades:':<20} {metrics['losing_trades']}",
            f"{'Win Rate:':<20} {metrics['win_rate']:.1f}%",
        ]
        for line in performance_lines:
            print(line)
        logger.debug(
            "Performance metrics: win_rate=%s%%, total_trades=%d",
            metrics["win_rate"],
            metrics["total_trades"],
        )

    def render_recent_trades(self, trades: list, *, limit: int = 5) -> None:
        print(f"\n📝 RECENT TRADES (Last {limit})")
        print("-" * 40)

        recent = trades[-limit:] if len(trades) > limit else trades
        if not recent:
            print("No trades executed")
            logger.debug("No trades executed")
            return

        print(f"{'Time':<20} {'Symbol':<10} {'Side':<6} {'Qty':<10} {'Price':<10} {'P&L'}")
        print("-" * 66)

        for trade in reversed(recent):
            time_str = trade.timestamp.strftime("%H:%M:%S")
            pnl_str = f"${trade.pnl:.2f}" if getattr(trade, "pnl", None) is not None else "-"
            print(
                f"{time_str:<20} {trade.symbol:<10} {trade.side:<6} "
                f"{trade.quantity:<10.6f} ${trade.price:<9.2f} {pnl_str}"
            )
        logger.debug("Displayed %d recent trades", len(recent))

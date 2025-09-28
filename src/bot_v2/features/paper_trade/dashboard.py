"""
Console dashboard for paper trading monitoring.
Displays positions, equity, metrics in a clean format.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ...config.path_registry import RESULTS_DIR


class PaperTradingDashboard:
    """Console dashboard for paper trading monitoring."""

    def __init__(self, engine: Any, refresh_interval: int = 5):
        """
        Initialize dashboard.

        Args:
            engine: PaperExecutionEngine instance
            refresh_interval: Seconds between updates
        """
        self.engine = engine
        self.refresh_interval = refresh_interval
        self.start_time = datetime.now()
        self.initial_equity = engine.initial_capital

    def clear_screen(self) -> None:
        """Clear the console screen."""
        os.system("clear" if os.name == "posix" else "cls")

    def format_currency(self, value: float) -> str:
        """Format value as currency."""
        return f"${value:,.2f}"

    def format_pct(self, value: float) -> str:
        """Format value as percentage with color."""
        if value > 0:
            return f"+{value:.2f}%"
        elif value < 0:
            return f"{value:.2f}%"
        else:
            return f"{value:.2f}%"

    def calculate_metrics(self) -> dict[str, Any]:
        """Calculate current metrics from engine state."""
        equity = self.engine.calculate_equity()

        # Calculate returns
        returns_pct = ((equity - self.initial_equity) / self.initial_equity) * 100

        # Calculate drawdown
        peak_equity = self.initial_equity
        for trade in self.engine.trades:
            # Rough estimate of equity after each trade
            peak_equity = max(peak_equity, equity)

        drawdown = 0.0
        if peak_equity > 0:
            drawdown = ((peak_equity - equity) / peak_equity) * 100

        # Calculate win rate from trades
        winning_trades = sum(
            1 for t in self.engine.trades if hasattr(t, "pnl") and t.pnl and t.pnl > 0
        )
        losing_trades = sum(
            1 for t in self.engine.trades if hasattr(t, "pnl") and t.pnl and t.pnl < 0
        )
        total_closed = winning_trades + losing_trades
        win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0.0

        # Calculate exposure
        positions_value = sum(
            pos.quantity * (pos.current_price if pos.current_price > 0 else pos.entry_price)
            for pos in self.engine.positions.values()
        )
        exposure_pct = (positions_value / equity * 100) if equity > 0 else 0.0

        return {
            "equity": equity,
            "cash": self.engine.cash,
            "positions_value": positions_value,
            "returns_pct": returns_pct,
            "drawdown_pct": drawdown,
            "win_rate": win_rate,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_trades": len(self.engine.trades),
            "exposure_pct": exposure_pct,
            "positions_count": len(self.engine.positions),
        }

    def print_header(self) -> None:
        """Print dashboard header."""
        print("=" * 80)
        print("                        PAPER TRADING DASHBOARD")
        print("=" * 80)
        print(f"Bot ID: {self.engine.bot_id}")
        print(f"Runtime: {datetime.now() - self.start_time}")
        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)

    def print_portfolio_summary(self, metrics: dict) -> None:
        """Print portfolio summary section."""
        print("\n📊 PORTFOLIO SUMMARY")
        print("-" * 40)

        # Two column layout
        print(
            f"{'Equity:':<20} {self.format_currency(metrics['equity']):<20} "
            f"{'Returns:':<15} {self.format_pct(metrics['returns_pct'])}"
        )

        print(
            f"{'Cash:':<20} {self.format_currency(metrics['cash']):<20} "
            f"{'Drawdown:':<15} {self.format_pct(metrics['drawdown_pct'])}"
        )

        print(
            f"{'Positions Value:':<20} {self.format_currency(metrics['positions_value']):<20} "
            f"{'Exposure:':<15} {self.format_pct(metrics['exposure_pct'])}"
        )

    def print_positions(self) -> None:
        """Print open positions."""
        print("\n📈 OPEN POSITIONS")
        print("-" * 40)

        if not self.engine.positions:
            print("No open positions")
        else:
            print(f"{'Symbol':<15} {'Qty':<12} {'Entry':<12} {'Current':<12} {'P&L':<12} {'P&L %'}")
            print("-" * 75)

            for symbol, pos in self.engine.positions.items():
                current = pos.current_price if pos.current_price > 0 else pos.entry_price
                pnl = (current - pos.entry_price) * pos.quantity
                pnl_pct = ((current / pos.entry_price) - 1) * 100 if pos.entry_price > 0 else 0

                print(
                    f"{symbol:<15} {pos.quantity:<12.6f} "
                    f"${pos.entry_price:<11.2f} ${current:<11.2f} "
                    f"${pnl:<11.2f} {self.format_pct(pnl_pct)}"
                )

    def print_performance(self, metrics: dict) -> None:
        """Print performance metrics."""
        print("\n📈 PERFORMANCE METRICS")
        print("-" * 40)

        print(f"{'Total Trades:':<20} {metrics['total_trades']}")
        print(f"{'Winning Trades:':<20} {metrics['winning_trades']}")
        print(f"{'Losing Trades:':<20} {metrics['losing_trades']}")
        print(f"{'Win Rate:':<20} {metrics['win_rate']:.1f}%")

    def print_recent_trades(self, limit: int = 5) -> None:
        """Print recent trades."""
        print(f"\n📝 RECENT TRADES (Last {limit})")
        print("-" * 40)

        recent = (
            self.engine.trades[-limit:] if len(self.engine.trades) > limit else self.engine.trades
        )

        if not recent:
            print("No trades executed")
        else:
            print(f"{'Time':<20} {'Symbol':<10} {'Side':<6} {'Qty':<10} {'Price':<10} {'P&L'}")
            print("-" * 66)

            for trade in reversed(recent):  # Show newest first
                time_str = trade.timestamp.strftime("%H:%M:%S")
                pnl_str = f"${trade.pnl:.2f}" if trade.pnl is not None else "-"

                print(
                    f"{time_str:<20} {trade.symbol:<10} {trade.side:<6} "
                    f"{trade.quantity:<10.6f} ${trade.price:<9.2f} {pnl_str}"
                )

    def display_once(self) -> None:
        """Display dashboard once without clearing screen."""
        metrics = self.calculate_metrics()

        self.print_header()
        self.print_portfolio_summary(metrics)
        self.print_positions()
        self.print_performance(metrics)
        self.print_recent_trades()
        print("\n" + "=" * 80)

    def display_continuous(self, duration: int | None = None) -> None:
        """
        Display dashboard continuously with refresh.

        Args:
            duration: Total seconds to run, or None for infinite
        """
        start = time.time()

        try:
            while True:
                self.clear_screen()
                self.display_once()

                # Check duration
                if duration and (time.time() - start) >= duration:
                    break

                # Show refresh countdown
                print(f"\nRefreshing in {self.refresh_interval} seconds... (Press Ctrl+C to stop)")
                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            print("\n\nDashboard stopped by user")

    def generate_html_summary(self, output_path: str | None = None) -> str:
        """
        Generate HTML summary report.

        Args:
            output_path: Path to save HTML file, or None to use default

        Returns:
            Path to saved HTML file
        """
        metrics = self.calculate_metrics()

        # Default path
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                RESULTS_DIR / f"paper_trading_summary_{self.engine.bot_id}_{timestamp}.html"
            )

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Paper Trading Summary - {self.engine.bot_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f9f9f9; padding: 15px; border-radius: 5px; }}
        .metric-label {{ color: #666; font-size: 12px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #4CAF50; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f5f5f5; }}
        .footer {{ text-align: center; color: #999; margin-top: 40px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Paper Trading Summary Report</h1>
        <p><strong>Bot ID:</strong> {self.engine.bot_id}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Portfolio Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Total Equity</div>
                <div class="metric-value">{self.format_currency(metrics['equity'])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Returns</div>
                <div class="metric-value {'positive' if metrics['returns_pct'] >= 0 else 'negative'}">{self.format_pct(metrics['returns_pct'])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{metrics['win_rate']:.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Cash Balance</div>
                <div class="metric-value">{self.format_currency(metrics['cash'])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Drawdown</div>
                <div class="metric-value negative">{self.format_pct(metrics['drawdown_pct'])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{metrics['total_trades']}</div>
            </div>
        </div>

        <h2>Open Positions</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Quantity</th>
                <th>Entry Price</th>
                <th>Current Price</th>
                <th>P&L</th>
                <th>P&L %</th>
            </tr>"""

        if self.engine.positions:
            for symbol, pos in self.engine.positions.items():
                current = pos.current_price if pos.current_price > 0 else pos.entry_price
                pnl = (current - pos.entry_price) * pos.quantity
                pnl_pct = ((current / pos.entry_price) - 1) * 100 if pos.entry_price > 0 else 0

                html += f"""
            <tr>
                <td>{symbol}</td>
                <td>{pos.quantity:.6f}</td>
                <td>${pos.entry_price:.2f}</td>
                <td>${current:.2f}</td>
                <td class="{'positive' if pnl >= 0 else 'negative'}">${pnl:.2f}</td>
                <td class="{'positive' if pnl_pct >= 0 else 'negative'}">{self.format_pct(pnl_pct)}</td>
            </tr>"""
        else:
            html += """
            <tr><td colspan="6" style="text-align: center; color: #999;">No open positions</td></tr>"""

        html += """
        </table>

        <h2>Recent Trades</h2>
        <table>
            <tr>
                <th>Time</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Quantity</th>
                <th>Price</th>
                <th>P&L</th>
            </tr>"""

        recent = self.engine.trades[-10:] if len(self.engine.trades) > 10 else self.engine.trades

        if recent:
            for trade in reversed(recent):
                pnl_str = f"${trade.pnl:.2f}" if trade.pnl is not None else "-"
                pnl_class = (
                    "positive"
                    if trade.pnl and trade.pnl > 0
                    else "negative" if trade.pnl and trade.pnl < 0 else ""
                )

                html += f"""
            <tr>
                <td>{trade.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                <td>{trade.symbol}</td>
                <td>{trade.side.upper()}</td>
                <td>{trade.quantity:.6f}</td>
                <td>${trade.price:.2f}</td>
                <td class="{pnl_class}">{pnl_str}</td>
            </tr>"""
        else:
            html += """
            <tr><td colspan="6" style="text-align: center; color: #999;">No trades executed</td></tr>"""

        html += f"""
        </table>

        <div class="footer">
            <p>Generated by Paper Trading System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""

        # Save HTML
        with open(output_path, "w") as f:
            f.write(html)

        return output_path

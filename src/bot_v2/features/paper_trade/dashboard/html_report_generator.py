"""HTML report generator for paper trading dashboard."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from bot_v2.config.path_registry import RESULTS_DIR
from bot_v2.features.paper_trade.dashboard.formatters import DashboardFormatter


class HTMLReportGenerator:
    """Generates HTML summary reports for paper trading sessions."""

    def __init__(self, formatter: DashboardFormatter) -> None:
        """
        Initialize HTML report generator.

        Args:
            formatter: Formatter for currency and percentage values
        """
        self.formatter = formatter

    def generate(
        self,
        *,
        bot_id: str,
        metrics: dict[str, Any],
        positions: dict[str, Any],
        trades: list,
        output_path: str | None = None,
    ) -> str:
        """
        Generate HTML summary report.

        Args:
            bot_id: Bot identifier
            metrics: Dashboard metrics dictionary
            positions: Open positions dictionary
            trades: List of trade objects
            output_path: Path to save HTML file, or None to use default

        Returns:
            Path to saved HTML file
        """
        # Generate default path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(RESULTS_DIR / f"paper_trading_summary_{bot_id}_{timestamp}.html")

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Build HTML content
        html = self._build_html(bot_id=bot_id, metrics=metrics, positions=positions, trades=trades)

        # Write to file
        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    def _build_html(
        self, *, bot_id: str, metrics: dict[str, Any], positions: dict[str, Any], trades: list
    ) -> str:
        """Build complete HTML document."""
        header = self._build_header(bot_id)
        metrics_section = self._build_metrics_section(metrics)
        positions_section = self._build_positions_section(positions)
        trades_section = self._build_trades_section(trades)
        footer = self._build_footer()

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Paper Trading Summary - {bot_id}</title>
    {self._get_styles()}
</head>
<body>
    <div class="container">
{header}
{metrics_section}
{positions_section}
{trades_section}
{footer}
    </div>
</body>
</html>"""

    def _get_styles(self) -> str:
        """Return CSS styles for the report."""
        return """<style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #666; margin-top: 30px; }
        .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }
        .metric { background: #f9f9f9; padding: 15px; border-radius: 5px; }
        .metric-label { color: #666; font-size: 12px; }
        .metric-value { font-size: 24px; font-weight: bold; margin-top: 5px; }
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th { background: #4CAF50; color: white; padding: 10px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f5f5f5; }
        .footer { text-align: center; color: #999; margin-top: 40px; font-size: 12px; }
    </style>"""

    def _build_header(self, bot_id: str) -> str:
        """Build report header section."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""        <h1>Paper Trading Summary Report</h1>
        <p><strong>Bot ID:</strong> {bot_id}</p>
        <p><strong>Generated:</strong> {timestamp}</p>"""

    def _build_metrics_section(self, metrics: dict[str, Any]) -> str:
        """Build portfolio metrics section."""
        returns_class = "positive" if metrics["returns_pct"] >= 0 else "negative"

        return f"""
        <h2>Portfolio Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Total Equity</div>
                <div class="metric-value">{self.formatter.currency_str(metrics['equity'])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Returns</div>
                <div class="metric-value {returns_class}">{self.formatter.percentage_str(metrics['returns_pct'])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{metrics['win_rate']:.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Cash Balance</div>
                <div class="metric-value">{self.formatter.currency_str(metrics['cash'])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Drawdown</div>
                <div class="metric-value negative">{self.formatter.percentage_str(metrics['drawdown_pct'])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{metrics['total_trades']}</div>
            </div>
        </div>"""

    def _build_positions_section(self, positions: dict[str, Any]) -> str:
        """Build open positions table section."""
        table_header = """
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

        if not positions:
            return (
                table_header
                + """
            <tr><td colspan="6" style="text-align: center; color: #999;">No open positions</td></tr>
        </table>"""
            )

        rows = []
        for symbol, pos in positions.items():
            current = pos.current_price if pos.current_price > 0 else pos.entry_price
            pnl = (current - pos.entry_price) * pos.quantity
            pnl_pct = ((current / pos.entry_price) - 1) * 100 if pos.entry_price > 0 else 0

            pnl_class = "positive" if pnl >= 0 else "negative"
            pnl_pct_class = "positive" if pnl_pct >= 0 else "negative"

            rows.append(
                f"""            <tr>
                <td>{symbol}</td>
                <td>{pos.quantity:.6f}</td>
                <td>${pos.entry_price:.2f}</td>
                <td>${current:.2f}</td>
                <td class="{pnl_class}">${pnl:.2f}</td>
                <td class="{pnl_pct_class}">{self.formatter.percentage_str(pnl_pct)}</td>
            </tr>"""
            )

        return table_header + "\n" + "\n".join(rows) + "\n        </table>"

    def _build_trades_section(self, trades: list) -> str:
        """Build recent trades table section."""
        table_header = """
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

        # Show last 10 trades
        recent = trades[-10:] if len(trades) > 10 else trades

        if not recent:
            return (
                table_header
                + """
            <tr><td colspan="6" style="text-align: center; color: #999;">No trades executed</td></tr>
        </table>"""
            )

        rows = []
        for trade in reversed(recent):
            pnl_str = f"${trade.pnl:.2f}" if trade.pnl is not None else "-"
            pnl_class = (
                "positive"
                if trade.pnl and trade.pnl > 0
                else "negative" if trade.pnl and trade.pnl < 0 else ""
            )

            rows.append(
                f"""            <tr>
                <td>{trade.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                <td>{trade.symbol}</td>
                <td>{trade.side.upper()}</td>
                <td>{trade.quantity:.6f}</td>
                <td>${trade.price:.2f}</td>
                <td class="{pnl_class}">{pnl_str}</td>
            </tr>"""
            )

        return table_header + "\n" + "\n".join(rows) + "\n        </table>"

    def _build_footer(self) -> str:
        """Build report footer section."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
        <div class="footer">
            <p>Generated by Paper Trading System | {timestamp}</p>
        </div>"""

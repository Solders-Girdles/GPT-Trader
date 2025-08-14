"""Live status dashboard for GPT-Trader."""

from __future__ import annotations

import random
import time
from datetime import datetime

import pytz
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

console = Console()


class TradingDashboard:
    """Live trading dashboard with real-time updates."""

    def __init__(self) -> None:
        self.console = console
        self.running = False
        self.update_interval = 1.0  # seconds

        # Mock data for demonstration
        self.positions = []
        self.recent_trades = []
        self.performance = {}
        self.alerts = []

    def run(self, mode: str = "live") -> None:
        """Run the dashboard."""
        self.running = True

        try:
            if mode == "live":
                self.run_live_dashboard()
            elif mode == "paper":
                self.run_paper_dashboard()
            elif mode == "backtest":
                self.run_backtest_dashboard()
            else:
                self.run_overview_dashboard()
        except KeyboardInterrupt:
            self.running = False
            self.console.print("\n[yellow]Dashboard stopped[/yellow]")

    def run_live_dashboard(self) -> None:
        """Run live trading dashboard."""
        layout = self.create_live_layout()

        with Live(layout, refresh_per_second=1, screen=True) as live:
            while self.running:
                try:
                    self.update_live_data()
                    layout = self.create_live_layout()
                    live.update(layout)
                    time.sleep(self.update_interval)
                except KeyboardInterrupt:
                    break

    def run_paper_dashboard(self) -> None:
        """Run paper trading dashboard."""
        layout = self.create_paper_layout()

        with Live(layout, refresh_per_second=1, screen=True) as live:
            while self.running:
                try:
                    self.update_paper_data()
                    layout = self.create_paper_layout()
                    live.update(layout)
                    time.sleep(self.update_interval)
                except KeyboardInterrupt:
                    break

    def run_backtest_dashboard(self) -> None:
        """Run backtest progress dashboard."""
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:

            # Main task
            main_task = progress.add_task("Overall Progress", total=100)

            # Sub-tasks
            data_task = progress.add_task("Loading Data", total=100)
            strategy_task = progress.add_task("Running Strategy", total=100)
            metrics_task = progress.add_task("Calculating Metrics", total=100)

            # Simulate progress
            for i in range(100):
                time.sleep(0.05)

                # Update sub-tasks
                if i < 30:
                    progress.update(data_task, advance=3.3)
                elif i < 70:
                    progress.update(strategy_task, advance=2.5)
                else:
                    progress.update(metrics_task, advance=3.3)

                # Update main task
                progress.update(main_task, advance=1)

                if not self.running:
                    break

    def run_overview_dashboard(self) -> None:
        """Run overview dashboard."""
        layout = self.create_overview_layout()

        with Live(layout, refresh_per_second=0.5, screen=True) as live:
            while self.running:
                try:
                    self.update_overview_data()
                    layout = self.create_overview_layout()
                    live.update(layout)
                    time.sleep(2)
                except KeyboardInterrupt:
                    break

    def create_live_layout(self) -> Layout:
        """Create live trading dashboard layout."""
        layout = Layout(name="root")

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        layout["header"].update(self.create_header("Live Trading"))
        layout["footer"].update(self.create_footer())

        # Main area split
        layout["main"].split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=1))

        # Left side - positions and trades
        layout["main"]["left"].split_column(
            Layout(self.create_positions_panel(), name="positions"),
            Layout(self.create_trades_panel(), name="trades"),
        )

        # Right side - performance and alerts
        layout["main"]["right"].split_column(
            Layout(self.create_performance_panel(), name="performance"),
            Layout(self.create_alerts_panel(), name="alerts"),
        )

        return layout

    def create_paper_layout(self) -> Layout:
        """Create paper trading dashboard layout."""
        layout = Layout(name="root")

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        layout["header"].update(self.create_header("Paper Trading"))
        layout["footer"].update(self.create_footer())

        # Main area
        layout["main"].split_column(
            Layout(self.create_paper_summary(), name="summary", size=8),
            Layout(name="details", ratio=1),
        )

        layout["main"]["details"].split_row(
            Layout(self.create_positions_panel(), name="positions"),
            Layout(self.create_paper_metrics(), name="metrics"),
        )

        return layout

    def create_overview_layout(self) -> Layout:
        """Create overview dashboard layout."""
        layout = Layout(name="root")

        layout.split_column(Layout(name="header", size=3), Layout(name="main", ratio=1))

        layout["header"].update(self.create_header("System Overview"))

        # Main grid
        layout["main"].split_column(
            Layout(name="top", size=10),
            Layout(name="middle", size=10),
            Layout(name="bottom", ratio=1),
        )

        layout["main"]["top"].split_row(
            Layout(self.create_market_status(), name="market"),
            Layout(self.create_system_status(), name="system"),
            Layout(self.create_account_status(), name="account"),
        )

        layout["main"]["middle"].update(self.create_strategy_status())
        layout["main"]["bottom"].update(self.create_activity_log())

        return layout

    def create_header(self, title: str) -> Panel:
        """Create dashboard header."""
        # Get current time
        ny_tz = pytz.timezone("America/New_York")
        now = datetime.now(ny_tz)
        time_str = now.strftime("%H:%M:%S %Z")

        # Market status
        market_open = self.is_market_open(now)
        market_status = "[green]● OPEN[/green]" if market_open else "[red]● CLOSED[/red]"

        # Header text
        header = Table(show_header=False, box=None, padding=0)
        header.add_column(justify="left")
        header.add_column(justify="center")
        header.add_column(justify="right")

        header.add_row(
            "[bold cyan]GPT-TRADER[/bold cyan]",
            f"[bold]{title}[/bold]",
            f"{market_status}  {time_str}",
        )

        return Panel(header, style="cyan", box="rounded")

    def create_footer(self) -> Panel:
        """Create dashboard footer."""
        footer_text = (
            "[dim]"
            "Q: Quit | P: Pause | R: Resume | H: Help | "
            "↑↓: Navigate | Enter: Select"
            "[/dim]"
        )
        return Panel(footer_text, style="dim", box="rounded")

    def create_positions_panel(self) -> Panel:
        """Create positions panel."""
        table = Table(title="Open Positions", show_header=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Side", justify="center")
        table.add_column("Qty", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")

        # Add mock positions
        if not self.positions:
            table.add_row("[dim]No open positions[/dim]", "", "", "", "", "", "")
        else:
            for pos in self.positions:
                pnl_style = "green" if pos["pnl"] > 0 else "red"
                table.add_row(
                    pos["symbol"],
                    "[green]LONG[/green]" if pos["side"] == "long" else "[red]SHORT[/red]",
                    str(pos["quantity"]),
                    f"${pos['entry']:.2f}",
                    f"${pos['current']:.2f}",
                    f"[{pnl_style}]${pos['pnl']:+.2f}[/{pnl_style}]",
                    f"[{pnl_style}]{pos['pnl_pct']:+.1f}%[/{pnl_style}]",
                )

        return Panel(table, border_style="blue")

    def create_trades_panel(self) -> Panel:
        """Create recent trades panel."""
        table = Table(title="Recent Trades", show_header=True)
        table.add_column("Time", style="dim")
        table.add_column("Symbol", style="cyan")
        table.add_column("Action")
        table.add_column("Qty", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Status")

        if not self.recent_trades:
            table.add_row("[dim]No recent trades[/dim]", "", "", "", "", "")
        else:
            for trade in self.recent_trades[-5:]:  # Show last 5 trades
                status_style = "green" if trade["status"] == "filled" else "yellow"
                action_style = "green" if trade["action"] == "BUY" else "red"

                table.add_row(
                    trade["time"],
                    trade["symbol"],
                    f"[{action_style}]{trade['action']}[/{action_style}]",
                    str(trade["quantity"]),
                    f"${trade['price']:.2f}",
                    f"[{status_style}]{trade['status']}[/{status_style}]",
                )

        return Panel(table, border_style="blue")

    def create_performance_panel(self) -> Panel:
        """Create performance metrics panel."""
        metrics = Table(title="Performance", show_header=False, box=None)
        metrics.add_column("Metric", style="cyan")
        metrics.add_column("Value", justify="right")

        # Performance data
        perf_data = self.performance or {
            "total_pnl": 0,
            "daily_pnl": 0,
            "win_rate": 0,
            "sharpe": 0,
            "max_dd": 0,
        }

        # Format and add metrics
        total_style = "green" if perf_data["total_pnl"] > 0 else "red"
        daily_style = "green" if perf_data["daily_pnl"] > 0 else "red"

        metrics.add_row(
            "Total P&L", f"[{total_style}]${perf_data['total_pnl']:+,.2f}[/{total_style}]"
        )
        metrics.add_row(
            "Daily P&L", f"[{daily_style}]${perf_data['daily_pnl']:+,.2f}[/{daily_style}]"
        )
        metrics.add_row("Win Rate", f"{perf_data['win_rate']:.1f}%")
        metrics.add_row("Sharpe Ratio", f"{perf_data['sharpe']:.2f}")
        metrics.add_row("Max Drawdown", f"[red]{perf_data['max_dd']:.1f}%[/red]")

        return Panel(metrics, border_style="green")

    def create_alerts_panel(self) -> Panel:
        """Create alerts panel."""
        alerts_list = Group()

        if not self.alerts:
            alerts_list = Text("[dim]No active alerts[/dim]", justify="center")
        else:
            alert_items = []
            for alert in self.alerts[-5:]:  # Show last 5 alerts
                icon = "⚠️ " if alert["level"] == "warning" else "🔴"
                color = "yellow" if alert["level"] == "warning" else "red"
                alert_items.append(Text(f"{icon} {alert['message']}", style=color))

            alerts_list = Group(*alert_items)

        return Panel(alerts_list, title="Alerts", border_style="yellow")

    def create_paper_summary(self) -> Panel:
        """Create paper trading summary."""
        summary = Table(show_header=False, box=None)
        summary.add_column("", style="cyan")
        summary.add_column("", justify="right")
        summary.add_column("", style="cyan")
        summary.add_column("", justify="right")

        summary.add_row("Account Value:", "$100,000.00", "Daily P&L:", "[green]+$1,234.56[/green]")
        summary.add_row("Buying Power:", "$45,678.90", "Open P&L:", "[red]-$234.50[/red]")
        summary.add_row("Positions:", "5", "Today's Trades:", "12")

        return Panel(summary, title="Paper Trading Summary", border_style="cyan")

    def create_paper_metrics(self) -> Panel:
        """Create paper trading metrics."""
        metrics = Table(title="Metrics", show_header=False, box=None)
        metrics.add_column("", style="cyan")
        metrics.add_column("", justify="right")

        metrics.add_row("Win Rate", "58.3%")
        metrics.add_row("Profit Factor", "1.85")
        metrics.add_row("Avg Win", "$456.78")
        metrics.add_row("Avg Loss", "$234.56")
        metrics.add_row("Best Trade", "[green]+$2,345.67[/green]")
        metrics.add_row("Worst Trade", "[red]-$1,234.56[/red]")

        return Panel(metrics, border_style="blue")

    def create_market_status(self) -> Panel:
        """Create market status panel."""
        table = Table(show_header=False, box=None)
        table.add_column("", style="cyan")
        table.add_column("", justify="right")

        # Market indicators
        table.add_row("S&P 500", "[green]4,532.10 ↑1.2%[/green]")
        table.add_row("NASDAQ", "[green]14,234.56 ↑1.5%[/green]")
        table.add_row("VIX", "[red]18.45 ↑5.2%[/red]")
        table.add_row("DXY", "102.34 →0.1%")

        return Panel(table, title="Market", border_style="blue")

    def create_system_status(self) -> Panel:
        """Create system status panel."""
        table = Table(show_header=False, box=None)
        table.add_column("", style="cyan")
        table.add_column("", justify="right")

        table.add_row("Status", "[green]● Running[/green]")
        table.add_row("Uptime", "2h 34m")
        table.add_row("CPU", "23%")
        table.add_row("Memory", "45%")

        return Panel(table, title="System", border_style="green")

    def create_account_status(self) -> Panel:
        """Create account status panel."""
        table = Table(show_header=False, box=None)
        table.add_column("", style="cyan")
        table.add_column("", justify="right")

        table.add_row("Balance", "$50,000")
        table.add_row("Equity", "$52,345")
        table.add_row("Margin", "$12,345")
        table.add_row("Available", "$37,655")

        return Panel(table, title="Account", border_style="yellow")

    def create_strategy_status(self) -> Panel:
        """Create strategy status panel."""
        table = Table(title="Active Strategies", show_header=True)
        table.add_column("Strategy", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Positions", justify="right")
        table.add_column("P&L Today", justify="right")
        table.add_column("P&L Total", justify="right")

        strategies = [
            ("trend_breakout", "active", 3, 234.56, 5678.90),
            ("demo_ma", "active", 2, -123.45, 3456.78),
            ("optimized_ma", "paused", 0, 0, 1234.56),
        ]

        for name, status, positions, daily, total in strategies:
            status_color = "green" if status == "active" else "yellow"
            daily_color = "green" if daily > 0 else "red" if daily < 0 else "white"
            total_color = "green" if total > 0 else "red" if total < 0 else "white"

            table.add_row(
                name,
                f"[{status_color}]● {status}[/{status_color}]",
                str(positions),
                f"[{daily_color}]${daily:+.2f}[/{daily_color}]",
                f"[{total_color}]${total:+.2f}[/{total_color}]",
            )

        return Panel(table, border_style="cyan")

    def create_activity_log(self) -> Panel:
        """Create activity log."""
        log_entries = [
            ("12:34:56", "info", "Strategy initialized: trend_breakout"),
            ("12:35:12", "trade", "BUY 100 AAPL @ $178.45"),
            ("12:36:23", "trade", "SELL 50 MSFT @ $412.30"),
            ("12:37:45", "warning", "High volatility detected in SPY"),
            ("12:38:02", "info", "Rebalancing portfolio"),
            ("12:39:15", "trade", "BUY 200 QQQ @ $385.20"),
        ]

        log_text = []
        for timestamp, level, message in log_entries:
            if level == "info":
                style = "dim"
            elif level == "trade":
                style = "cyan"
            elif level == "warning":
                style = "yellow"
            else:
                style = "white"

            log_text.append(f"[dim]{timestamp}[/dim] [{style}]{message}[/{style}]")

        return Panel("\n".join(log_text), title="Activity Log", border_style="dim")

    def is_market_open(self, now: datetime) -> bool:
        """Check if market is open."""
        if now.weekday() >= 5:  # Weekend
            return False

        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)

        return market_open <= now <= market_close

    def update_live_data(self) -> None:
        """Update live trading data (mock)."""
        # Simulate position updates
        if random.random() > 0.7 and len(self.positions) < 5:
            self.positions.append(
                {
                    "symbol": random.choice(["AAPL", "MSFT", "GOOGL", "AMZN"]),
                    "side": "long",
                    "quantity": random.randint(10, 100),
                    "entry": random.uniform(100, 500),
                    "current": random.uniform(100, 500),
                    "pnl": random.uniform(-1000, 2000),
                    "pnl_pct": random.uniform(-5, 10),
                }
            )

        # Update existing positions
        for pos in self.positions:
            pos["current"] += random.uniform(-2, 2)
            pos["pnl"] = (pos["current"] - pos["entry"]) * pos["quantity"]
            pos["pnl_pct"] = (pos["current"] / pos["entry"] - 1) * 100

        # Simulate trades
        if random.random() > 0.8:
            self.recent_trades.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "symbol": random.choice(["AAPL", "MSFT", "GOOGL"]),
                    "action": random.choice(["BUY", "SELL"]),
                    "quantity": random.randint(10, 100),
                    "price": random.uniform(100, 500),
                    "status": "filled",
                }
            )

        # Update performance
        self.performance = {
            "total_pnl": sum(p["pnl"] for p in self.positions),
            "daily_pnl": random.uniform(-500, 1000),
            "win_rate": random.uniform(40, 70),
            "sharpe": random.uniform(0.5, 2.0),
            "max_dd": random.uniform(5, 20),
        }

    def update_paper_data(self) -> None:
        """Update paper trading data (mock)."""
        self.update_live_data()  # Use same mock data

    def update_overview_data(self) -> None:
        """Update overview data (mock)."""
        # Add random alerts
        if random.random() > 0.9:
            levels = ["info", "warning", "critical"]
            messages = [
                "Strategy parameter update",
                "High volatility detected",
                "Risk limit approaching",
                "Connection restored",
            ]

            self.alerts.append(
                {
                    "level": random.choice(levels),
                    "message": random.choice(messages),
                    "time": datetime.now(),
                }
            )


def run_dashboard(mode: str = "overview") -> None:
    """Run the trading dashboard."""
    dashboard = TradingDashboard()
    dashboard.run(mode)


if __name__ == "__main__":
    run_dashboard()

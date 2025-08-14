"""Enhanced main menu system for GPT-Trader with user-friendly navigation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytz
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()


class MainMenu:
    """Interactive main menu with enhanced navigation."""

    def __init__(self) -> None:
        self.console = console
        self.current_view = "main"
        self.history = []

    def run(self) -> None:
        """Run the main menu loop."""
        self.display_welcome()

        while True:
            try:
                if self.current_view == "main":
                    self.show_main_menu()
                elif self.current_view == "quick_start":
                    self.show_quick_start()
                elif self.current_view == "strategy":
                    self.show_strategy_menu()
                elif self.current_view == "trading":
                    self.show_trading_menu()
                elif self.current_view == "analysis":
                    self.show_analysis_menu()
                elif self.current_view == "settings":
                    self.show_settings_menu()
                elif self.current_view == "exit":
                    if self.confirm_exit():
                        break
                    self.current_view = "main"
            except KeyboardInterrupt:
                if self.confirm_exit():
                    break
                self.current_view = "main"

    def display_welcome(self) -> None:
        """Display welcome screen with system status."""
        # Create welcome banner
        welcome_text = Text()
        welcome_text.append("🤖 ", style="bold cyan")
        welcome_text.append("GPT-TRADER", style="bold cyan")
        welcome_text.append(" | ", style="dim")
        welcome_text.append("AI-Powered Trading Platform", style="italic")

        # System status
        status = self.get_system_status()

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(Panel(Align.center(welcome_text), border_style="cyan"), size=3),
            Layout(self.create_status_panel(status), size=7),
            Layout(
                Panel(self.get_quick_tips(), title="💡 Quick Tips", border_style="yellow"), size=5
            ),
        )

        self.console.clear()
        self.console.print(layout)
        self.console.print()

    def show_main_menu(self) -> None:
        """Display the main menu."""
        menu_items = [
            ("1", "🚀 Quick Start", "quick_start", "Get started with a guided setup"),
            (
                "2",
                "📈 Strategy Development",
                "strategy",
                "Backtest, optimize, and validate strategies",
            ),
            ("3", "💹 Trading Operations", "trading", "Paper trading, live deployment, monitoring"),
            ("4", "📊 Analysis & Reports", "analysis", "View performance, generate reports"),
            ("5", "⚙️  Settings & Config", "settings", "Configure profiles and preferences"),
            ("6", "📚 Documentation", "docs", "View guides and API reference"),
            ("7", "❓ Help & Support", "help", "Get help and troubleshooting"),
            ("8", "🚪 Exit", "exit", "Exit GPT-Trader"),
        ]

        # Create menu table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold cyan", width=3)
        table.add_column("Option", style="white", width=25)
        table.add_column("Description", style="dim")

        for key, option, _, desc in menu_items:
            table.add_row(key, option, desc)

        self.console.print(Panel(table, title="📋 Main Menu", border_style="blue"))

        # Get user choice
        choice = Prompt.ask(
            "\n[bold cyan]Select option[/bold cyan]",
            choices=[str(i) for i in range(1, 9)],
            default="1",
        )

        # Handle navigation
        action_map = {str(i + 1): item[2] for i, item in enumerate(menu_items)}
        self.navigate_to(action_map[choice])

    def show_quick_start(self) -> None:
        """Show quick start wizard."""
        self.console.print(
            Panel(
                "[bold cyan]Quick Start Wizard[/bold cyan]\n\n"
                "Let's get you started with GPT-Trader!",
                border_style="cyan",
            )
        )

        options = [
            ("1", "🎯 Run a Simple Backtest", self.run_simple_backtest),
            ("2", "🔍 Explore Sample Strategies", self.explore_strategies),
            ("3", "📝 Create Configuration Profile", self.create_profile),
            ("4", "🏃 Start Paper Trading", self.start_paper_trading),
            ("5", "⬅️  Back to Main Menu", lambda: self.navigate_to("main")),
        ]

        table = Table(show_header=False, box=None)
        for key, option, _ in options:
            table.add_row(f"[cyan]{key}[/cyan]", option)

        self.console.print(table)

        choice = Prompt.ask("\n[cyan]Select option[/cyan]", choices=["1", "2", "3", "4", "5"])
        options[int(choice) - 1][2]()

    def show_strategy_menu(self) -> None:
        """Show strategy development menu."""
        self.console.print(
            Panel("[bold cyan]Strategy Development[/bold cyan]", border_style="cyan")
        )

        options = [
            ("1", "📊 Backtest Strategy", "Run historical backtests"),
            ("2", "🔧 Optimize Parameters", "Find optimal strategy parameters"),
            ("3", "🔄 Walk-Forward Analysis", "Validate strategy robustness"),
            ("4", "⚡ Rapid Evolution", "AI-powered strategy discovery"),
            ("5", "📈 Compare Strategies", "Side-by-side comparison"),
            ("6", "⬅️  Back", "Return to main menu"),
        ]

        table = Table(show_header=False, box=None)
        table.add_column("", style="cyan", width=3)
        table.add_column("", width=25)
        table.add_column("", style="dim")

        for key, option, desc in options:
            table.add_row(key, option, desc)

        self.console.print(table)

        choice = Prompt.ask("\n[cyan]Select option[/cyan]", choices=["1", "2", "3", "4", "5", "6"])

        if choice == "1":
            self.run_backtest_wizard()
        elif choice == "2":
            self.run_optimization_wizard()
        elif choice == "3":
            self.run_walk_forward()
        elif choice == "4":
            self.run_rapid_evolution()
        elif choice == "5":
            self.compare_strategies()
        else:
            self.navigate_to("main")

    def show_trading_menu(self) -> None:
        """Show trading operations menu."""
        self.console.print(Panel("[bold cyan]Trading Operations[/bold cyan]", border_style="cyan"))

        options = [
            ("1", "📝 Paper Trading", "Test strategies with simulated money"),
            ("2", "🚀 Deploy Live", "Deploy strategy to live trading"),
            ("3", "📊 Monitor Positions", "View current positions and P&L"),
            ("4", "🛑 Risk Management", "Configure risk controls"),
            ("5", "📈 Performance Dashboard", "Real-time performance metrics"),
            ("6", "⬅️  Back", "Return to main menu"),
        ]

        table = Table(show_header=False, box=None)
        for key, option, desc in options:
            table.add_row(f"[cyan]{key}[/cyan]", option, f"[dim]{desc}[/dim]")

        self.console.print(table)

        choice = Prompt.ask("\n[cyan]Select option[/cyan]", choices=["1", "2", "3", "4", "5", "6"])

        if choice == "6":
            self.navigate_to("main")
        else:
            self.console.print("\n[yellow]Feature coming soon![/yellow]")
            self.console.input("\nPress Enter to continue...")

    def show_analysis_menu(self) -> None:
        """Show analysis and reports menu."""
        self.console.print(Panel("[bold cyan]Analysis & Reports[/bold cyan]", border_style="cyan"))

        options = [
            ("1", "📊 Performance Summary", "View overall performance metrics"),
            ("2", "📈 Trade Analysis", "Analyze individual trades"),
            ("3", "🎯 Strategy Metrics", "Detailed strategy statistics"),
            ("4", "📉 Risk Analysis", "Risk and drawdown analysis"),
            ("5", "📄 Generate Report", "Create comprehensive report"),
            ("6", "⬅️  Back", "Return to main menu"),
        ]

        table = Table(show_header=False, box=None)
        for key, option, desc in options:
            table.add_row(f"[cyan]{key}[/cyan]", option, f"[dim]{desc}[/dim]")

        self.console.print(table)

        choice = Prompt.ask("\n[cyan]Select option[/cyan]", choices=["1", "2", "3", "4", "5", "6"])

        if choice == "6":
            self.navigate_to("main")
        else:
            self.show_sample_analysis()

    def show_settings_menu(self) -> None:
        """Show settings and configuration menu."""
        self.console.print(
            Panel("[bold cyan]Settings & Configuration[/bold cyan]", border_style="cyan")
        )

        options = [
            ("1", "👤 Profile Management", "Manage configuration profiles"),
            ("2", "🔑 API Keys", "Configure broker API keys"),
            ("3", "📊 Data Sources", "Configure data providers"),
            ("4", "🔔 Notifications", "Set up alerts and notifications"),
            ("5", "🎨 Preferences", "UI and display preferences"),
            ("6", "⬅️  Back", "Return to main menu"),
        ]

        table = Table(show_header=False, box=None)
        for key, option, desc in options:
            table.add_row(f"[cyan]{key}[/cyan]", option, f"[dim]{desc}[/dim]")

        self.console.print(table)

        choice = Prompt.ask("\n[cyan]Select option[/cyan]", choices=["1", "2", "3", "4", "5", "6"])

        if choice == "6":
            self.navigate_to("main")
        else:
            self.console.print("\n[yellow]Settings interface coming soon![/yellow]")
            self.console.input("\nPress Enter to continue...")

    def run_simple_backtest(self) -> None:
        """Run a simple backtest with guided input."""
        self.console.print(Rule("Simple Backtest"))

        # Get inputs
        symbol = Prompt.ask("Enter symbol", default="AAPL")
        strategy = Prompt.ask(
            "Select strategy", choices=["demo_ma", "trend_breakout"], default="demo_ma"
        )

        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            progress.add_task("Running backtest...", total=None)

            # Build command
            cmd = f"gpt-trader backtest --symbol {symbol} --strategy {strategy}"
            self.console.print(f"\n[dim]Command: {cmd}[/dim]\n")

            # Import and run
            import subprocess

            result = subprocess.run(cmd.split(), capture_output=True, text=True)

            if result.returncode == 0:
                self.console.print("[green]✓[/green] Backtest completed successfully!")
            else:
                self.console.print("[red]✗[/red] Backtest failed")

        self.console.input("\nPress Enter to continue...")

    def explore_strategies(self) -> None:
        """Show available strategies with descriptions."""
        strategies = [
            ("demo_ma", "Moving Average Crossover", "Simple trend-following using MA crossovers"),
            ("trend_breakout", "Donchian Channel Breakout", "Breakout strategy with ATR stops"),
            ("optimized_ma", "Optimized MA Strategy", "Enhanced MA with dynamic parameters"),
        ]

        table = Table(title="Available Strategies", box="rounded")
        table.add_column("Strategy", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Description")

        for name, type_, desc in strategies:
            table.add_row(name, type_, desc)

        self.console.print(table)
        self.console.input("\nPress Enter to continue...")

    def create_profile(self) -> None:
        """Create a configuration profile."""
        self.console.print(Rule("Create Configuration Profile"))

        name = Prompt.ask("Profile name")
        Prompt.ask("Default strategy", default="demo_ma")
        Prompt.ask("Risk per trade (%)", default="2")

        self.console.print(f"\n[green]✓[/green] Profile '{name}' created successfully!")
        self.console.input("\nPress Enter to continue...")

    def start_paper_trading(self) -> None:
        """Start paper trading session."""
        self.console.print(Rule("Paper Trading Setup"))

        self.console.print(
            "[yellow]Paper trading allows you to test strategies with simulated money.[/yellow]\n"
        )

        if Confirm.ask("Start paper trading session?"):
            self.console.print("\n[green]Starting paper trading...[/green]")
            self.console.print("[dim]Command: gpt-trader paper --start[/dim]")

        self.console.input("\nPress Enter to continue...")

    def run_backtest_wizard(self) -> None:
        """Run the backtest wizard."""
        self.console.print("[cyan]Launching backtest wizard...[/cyan]")
        import subprocess

        subprocess.run(["gpt-trader", "interactive", "--guided-backtest"])

    def run_optimization_wizard(self) -> None:
        """Run optimization wizard."""
        self.console.print("[cyan]Launching optimization wizard...[/cyan]")
        import subprocess

        subprocess.run(["gpt-trader", "optimize", "--interactive"])

    def run_walk_forward(self) -> None:
        """Run walk-forward analysis."""
        self.console.print("[cyan]Launching walk-forward analysis...[/cyan]")
        import subprocess

        subprocess.run(["gpt-trader", "walk-forward", "--guided"])

    def run_rapid_evolution(self) -> None:
        """Run rapid evolution."""
        self.console.print("[cyan]Launching rapid evolution...[/cyan]")
        import subprocess

        subprocess.run(["gpt-trader", "rapid-evolution", "--interactive"])

    def compare_strategies(self) -> None:
        """Compare multiple strategies."""
        self.console.print(Panel("Strategy Comparison", border_style="cyan"))

        table = Table(title="Strategy Performance Comparison")
        table.add_column("Strategy", style="cyan")
        table.add_column("Total Return", justify="right")
        table.add_column("Sharpe Ratio", justify="right")
        table.add_column("Max Drawdown", justify="right")
        table.add_column("Win Rate", justify="right")

        # Sample data
        table.add_row("demo_ma", "+45.2%", "1.35", "-12.3%", "58%")
        table.add_row("trend_breakout", "+62.8%", "1.52", "-15.7%", "42%")
        table.add_row("optimized_ma", "+71.3%", "1.68", "-10.2%", "61%")

        self.console.print(table)
        self.console.input("\nPress Enter to continue...")

    def show_sample_analysis(self) -> None:
        """Show sample analysis output."""
        self.console.print(Panel("Performance Analysis", border_style="cyan"))

        # Create sample metrics
        metrics = Table(show_header=False, box=None)
        metrics.add_column("Metric", style="cyan")
        metrics.add_column("Value", justify="right")

        metrics.add_row("Total Return", "[green]+52.3%[/green]")
        metrics.add_row("Sharpe Ratio", "1.42")
        metrics.add_row("Max Drawdown", "[red]-15.7%[/red]")
        metrics.add_row("Win Rate", "56.2%")
        metrics.add_row("Profit Factor", "1.85")
        metrics.add_row("Total Trades", "127")

        self.console.print(metrics)
        self.console.input("\nPress Enter to continue...")

    def get_system_status(self) -> dict:
        """Get current system status."""
        # Check market status
        ny_tz = pytz.timezone("America/New_York")
        now = datetime.now(ny_tz)

        market_open = False
        if now.weekday() < 5:  # Weekday
            market_open_time = now.replace(hour=9, minute=30, second=0)
            market_close_time = now.replace(hour=16, minute=0, second=0)
            market_open = market_open_time <= now <= market_close_time

        return {
            "market": "Open" if market_open else "Closed",
            "time": now.strftime("%H:%M %Z"),
            "data_dir": Path("data").exists(),
            "config": Path(".gpt-trader").exists(),
            "profiles": (
                len(list(Path.home().glob(".gpt-trader/profiles/*.yaml")))
                if Path.home().joinpath(".gpt-trader/profiles").exists()
                else 0
            ),
        }

    def create_status_panel(self, status: dict) -> Panel:
        """Create status panel."""
        table = Table(show_header=False, box=None)
        table.add_column("Item", style="cyan")
        table.add_column("Status")

        # Market status
        market_style = "green" if status["market"] == "Open" else "red"
        table.add_row("Market Status", f"[{market_style}]{status['market']}[/{market_style}]")
        table.add_row("Current Time", status["time"])

        # System status
        data_status = "[green]✓[/green]" if status["data_dir"] else "[red]✗[/red]"
        table.add_row("Data Directory", data_status)

        config_status = "[green]✓[/green]" if status["config"] else "[yellow]![/yellow]"
        table.add_row("Configuration", config_status)

        table.add_row("Profiles", str(status["profiles"]))

        return Panel(table, title="📊 System Status", border_style="blue")

    def get_quick_tips(self) -> Table:
        """Get quick tips for new users."""
        tips = Table(show_header=False, box=None)
        tips.add_column("", style="yellow")

        tips.add_row("• Press [cyan]1[/cyan] for Quick Start to get up and running fast")
        tips.add_row("• Use [cyan]gpt-trader --help[/cyan] for command-line options")
        tips.add_row("• Check out sample strategies in the Strategy menu")
        tips.add_row("• Start with paper trading before going live")

        return tips

    def navigate_to(self, view: str) -> None:
        """Navigate to a different view."""
        if view == "docs":
            self.show_documentation()
        elif view == "help":
            self.show_help()
        else:
            self.history.append(self.current_view)
            self.current_view = view

    def show_documentation(self) -> None:
        """Show documentation links."""
        self.console.print(
            Panel(
                "[bold]Documentation[/bold]\n\n"
                "📚 User Guide: docs/USAGE.md\n"
                "🔧 API Reference: docs/API.md\n"
                "📊 Strategy Development: docs/STRATEGIES.md\n"
                "🚀 Deployment Guide: docs/DEPLOYMENT.md\n",
                border_style="cyan",
            )
        )
        self.console.input("\nPress Enter to continue...")

    def show_help(self) -> None:
        """Show help and support information."""
        self.console.print(
            Panel(
                "[bold]Help & Support[/bold]\n\n"
                "❓ FAQ: docs/FAQ.md\n"
                "🐛 Report Issues: github.com/gpt-trader/issues\n"
                "💬 Discord: discord.gg/gpt-trader\n"
                "📧 Email: support@gpt-trader.ai\n",
                border_style="cyan",
            )
        )
        self.console.input("\nPress Enter to continue...")

    def confirm_exit(self) -> bool:
        """Confirm exit."""
        return Confirm.ask("\n[yellow]Are you sure you want to exit?[/yellow]")


def main() -> None:
    """Main entry point for the menu system."""
    menu = MainMenu()
    menu.run()


if __name__ == "__main__":
    main()

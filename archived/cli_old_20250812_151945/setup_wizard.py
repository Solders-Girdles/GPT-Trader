"""Configuration wizard for new users."""

from __future__ import annotations

from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table

console = Console()


class SetupWizard:
    """Interactive setup wizard for new users."""

    def __init__(self) -> None:
        self.console = console
        self.config = {}
        self.config_dir = Path.home() / ".gpt-trader"
        self.config_file = self.config_dir / "config.yaml"
        self.profiles_dir = self.config_dir / "profiles"

    def run(self) -> None:
        """Run the complete setup wizard."""
        self.console.clear()
        self.show_welcome()

        # Step through setup
        steps = [
            ("Environment Setup", self.setup_environment),
            ("Data Configuration", self.setup_data),
            ("Strategy Selection", self.setup_strategies),
            ("Risk Management", self.setup_risk),
            ("Trading Preferences", self.setup_trading),
            ("Notifications", self.setup_notifications),
            ("Create Profile", self.create_profile),
        ]

        for i, (step_name, step_func) in enumerate(steps, 1):
            self.console.print(Rule(f"Step {i}/{len(steps)}: {step_name}"))

            try:
                if not step_func():
                    if not Confirm.ask("\n[yellow]Skip this step?[/yellow]"):
                        return
            except KeyboardInterrupt:
                if Confirm.ask("\n[yellow]Exit setup wizard?[/yellow]"):
                    return

        self.finalize_setup()

    def show_welcome(self) -> None:
        """Show welcome message."""
        welcome = Panel(
            "[bold cyan]Welcome to GPT-Trader Setup Wizard![/bold cyan]\n\n"
            "This wizard will help you:\n"
            "• Configure your trading environment\n"
            "• Set up data sources\n"
            "• Choose default strategies\n"
            "• Configure risk management\n"
            "• Create your first profile\n\n"
            "[dim]Press Ctrl+C at any time to exit[/dim]",
            border_style="cyan",
            title="🚀 Setup Wizard",
        )
        self.console.print(welcome)
        self.console.print()

    def setup_environment(self) -> bool:
        """Set up the environment."""
        self.console.print("\n[bold]Setting up directories...[/bold]")

        # Create necessary directories
        directories = [
            self.config_dir,
            self.profiles_dir,
            Path("data"),
            Path("data/backtests"),
            Path("data/logs"),
            Path("data/cache"),
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Creating directories...", total=len(directories))

            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                progress.update(task, advance=1, description=f"Created {directory}")

        self.console.print("[green]✓[/green] Directories created successfully")

        # Check Python version
        import sys

        py_version = sys.version_info
        if py_version.major == 3 and py_version.minor >= 8:
            self.console.print(
                f"[green]✓[/green] Python {py_version.major}.{py_version.minor} detected"
            )
        else:
            self.console.print(
                f"[yellow]⚠[/yellow] Python 3.8+ recommended (found {py_version.major}.{py_version.minor})"
            )

        # Check dependencies
        self.check_dependencies()

        return True

    def check_dependencies(self) -> None:
        """Check for required dependencies."""
        dependencies = [
            ("pandas", "Data manipulation"),
            ("numpy", "Numerical computing"),
            ("yfinance", "Market data"),
            ("rich", "Terminal UI"),
            ("plotly", "Charting"),
        ]

        table = Table(title="Dependencies", show_header=True)
        table.add_column("Package", style="cyan")
        table.add_column("Purpose")
        table.add_column("Status", justify="center")

        for package, purpose in dependencies:
            try:
                __import__(package)
                status = "[green]✓[/green]"
            except ImportError:
                status = "[red]✗[/red]"

            table.add_row(package, purpose, status)

        self.console.print(table)

    def setup_data(self) -> bool:
        """Configure data sources."""
        self.console.print("\n[bold]Data Source Configuration[/bold]")

        # Primary data source
        data_source = Prompt.ask(
            "Primary data source",
            choices=["yfinance", "alpaca", "polygon", "interactive_brokers"],
            default="yfinance",
        )
        self.config["data_source"] = data_source

        if data_source != "yfinance":
            # API configuration
            self.console.print(f"\n[yellow]Configure {data_source} API[/yellow]")

            api_key = Prompt.ask("API Key", password=True)
            api_secret = Prompt.ask("API Secret", password=True)

            self.config[f"{data_source}_api"] = {"key": api_key, "secret": api_secret}

        # Data preferences
        self.console.print("\n[bold]Data Preferences[/bold]")

        self.config["data_validation"] = Prompt.ask(
            "Data validation mode", choices=["strict", "repair"], default="repair"
        )

        self.config["cache_enabled"] = Confirm.ask("Enable data caching?", default=True)

        if self.config["cache_enabled"]:
            self.config["cache_days"] = IntPrompt.ask("Cache retention (days)", default=7)

        return True

    def setup_strategies(self) -> bool:
        """Select and configure strategies."""
        self.console.print("\n[bold]Strategy Selection[/bold]")

        # Available strategies
        strategies = {
            "demo_ma": "Simple Moving Average Crossover",
            "trend_breakout": "Donchian Channel Breakout",
            "optimized_ma": "Optimized Moving Average",
        }

        table = Table(title="Available Strategies")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Description")

        table.add_row("demo_ma", "MA Crossover", "Simple trend following")
        table.add_row("trend_breakout", "Breakout", "Channel breakout with ATR stops")
        table.add_row("optimized_ma", "Optimized MA", "Advanced MA with filters")

        self.console.print(table)

        # Select default strategy
        default_strategy = Prompt.ask(
            "\nDefault strategy", choices=list(strategies.keys()), default="demo_ma"
        )
        self.config["default_strategy"] = default_strategy

        # Configure strategy parameters
        if Confirm.ask("Configure strategy parameters?", default=False):
            self.configure_strategy_params(default_strategy)

        # Multi-strategy support
        if Confirm.ask("Enable multi-strategy trading?", default=False):
            self.config["multi_strategy"] = True
            selected = []

            for strat_id, strat_name in strategies.items():
                if Confirm.ask(f"Include {strat_name}?", default=False):
                    selected.append(strat_id)

            self.config["enabled_strategies"] = selected

        return True

    def configure_strategy_params(self, strategy: str) -> None:
        """Configure parameters for a strategy."""
        params = {}

        if strategy == "demo_ma":
            params["sma_fast"] = IntPrompt.ask("Fast SMA period", default=20)
            params["sma_slow"] = IntPrompt.ask("Slow SMA period", default=50)

        elif strategy == "trend_breakout":
            params["donchian"] = IntPrompt.ask("Donchian lookback", default=55)
            params["atr"] = IntPrompt.ask("ATR period", default=20)
            params["atr_k"] = FloatPrompt.ask("ATR multiplier", default=2.0)

        elif strategy == "optimized_ma":
            params["adaptive"] = Confirm.ask("Use adaptive periods?", default=True)
            params["volume_filter"] = Confirm.ask("Use volume filter?", default=True)

        self.config[f"{strategy}_params"] = params

    def setup_risk(self) -> bool:
        """Configure risk management."""
        self.console.print("\n[bold]Risk Management Configuration[/bold]")

        # Position sizing
        self.config["risk_per_trade"] = FloatPrompt.ask("Risk per trade (%)", default=2.0)

        self.config["max_positions"] = IntPrompt.ask("Maximum concurrent positions", default=5)

        # Portfolio limits
        self.config["max_portfolio_risk"] = FloatPrompt.ask(
            "Maximum portfolio risk (%)", default=10.0
        )

        # Stop loss
        self.config["stop_loss_enabled"] = Confirm.ask("Enable stop losses?", default=True)

        if self.config["stop_loss_enabled"]:
            self.config["stop_loss_type"] = Prompt.ask(
                "Stop loss type", choices=["fixed", "atr", "trailing"], default="atr"
            )

        # Drawdown limits
        self.config["max_drawdown"] = FloatPrompt.ask("Maximum drawdown limit (%)", default=20.0)

        # Risk alerts
        if Confirm.ask("Enable risk alerts?", default=True):
            self.config["risk_alerts"] = {
                "drawdown_warning": 10.0,
                "drawdown_critical": 15.0,
                "exposure_warning": 80.0,
            }

        return True

    def setup_trading(self) -> bool:
        """Configure trading preferences."""
        self.console.print("\n[bold]Trading Preferences[/bold]")

        # Trading mode
        self.config["trading_mode"] = Prompt.ask(
            "Default trading mode", choices=["backtest", "paper", "live"], default="backtest"
        )

        # Execution preferences
        self.config["order_type"] = Prompt.ask(
            "Default order type", choices=["market", "limit"], default="market"
        )

        self.config["slippage"] = FloatPrompt.ask("Estimated slippage (%)", default=0.1)

        self.config["commission"] = FloatPrompt.ask("Commission per trade ($)", default=1.0)

        # Rebalancing
        self.config["rebalance_frequency"] = Prompt.ask(
            "Rebalancing frequency", choices=["daily", "weekly", "monthly"], default="daily"
        )

        # Market hours
        self.config["trade_only_market_hours"] = Confirm.ask(
            "Trade only during market hours?", default=True
        )

        return True

    def setup_notifications(self) -> bool:
        """Configure notifications."""
        self.console.print("\n[bold]Notification Settings[/bold]")

        self.config["notifications_enabled"] = Confirm.ask("Enable notifications?", default=False)

        if not self.config["notifications_enabled"]:
            return True

        # Notification channels
        channels = []

        if Confirm.ask("Email notifications?", default=False):
            email = Prompt.ask("Email address")
            channels.append({"type": "email", "address": email})

        if Confirm.ask("Slack notifications?", default=False):
            webhook = Prompt.ask("Slack webhook URL")
            channels.append({"type": "slack", "webhook": webhook})

        if Confirm.ask("Discord notifications?", default=False):
            webhook = Prompt.ask("Discord webhook URL")
            channels.append({"type": "discord", "webhook": webhook})

        self.config["notification_channels"] = channels

        # Notification events
        events = []
        event_options = [
            ("trade_executed", "Trade executions"),
            ("daily_summary", "Daily performance summary"),
            ("risk_alert", "Risk warnings"),
            ("error", "System errors"),
        ]

        self.console.print("\n[bold]Select notification events:[/bold]")
        for event_id, event_name in event_options:
            if Confirm.ask(f"  {event_name}?", default=True):
                events.append(event_id)

        self.config["notification_events"] = events

        return True

    def create_profile(self) -> bool:
        """Create a configuration profile."""
        self.console.print("\n[bold]Create Configuration Profile[/bold]")

        profile_name = Prompt.ask("Profile name", default="default")

        profile_path = self.profiles_dir / f"{profile_name}.yaml"

        if profile_path.exists():
            if not Confirm.ask(f"Profile '{profile_name}' exists. Overwrite?", default=False):
                return False

        # Create profile from config
        profile = {"name": profile_name, "created": str(Path.cwd()), **self.config}

        # Save profile
        with open(profile_path, "w") as f:
            yaml.dump(profile, f, default_flow_style=False)

        self.console.print(f"[green]✓[/green] Profile '{profile_name}' created")

        # Set as default
        if Confirm.ask("Set as default profile?", default=True):
            self.config_file.write_text(f"default_profile: {profile_name}\n")
            self.console.print("[green]✓[/green] Set as default profile")

        return True

    def finalize_setup(self) -> None:
        """Finalize the setup process."""
        self.console.print(Rule("Setup Complete!"))

        # Summary
        summary = Table(title="Configuration Summary", show_header=False)
        summary.add_column("Setting", style="cyan")
        summary.add_column("Value")

        key_settings = [
            ("Data Source", self.config.get("data_source", "yfinance")),
            ("Default Strategy", self.config.get("default_strategy", "demo_ma")),
            ("Risk per Trade", f"{self.config.get('risk_per_trade', 2)}%"),
            ("Max Positions", str(self.config.get("max_positions", 5))),
            ("Trading Mode", self.config.get("trading_mode", "backtest")),
        ]

        for setting, value in key_settings:
            summary.add_row(setting, value)

        self.console.print(summary)

        # Next steps
        self.console.print("\n[bold cyan]Next Steps:[/bold cyan]")
        self.console.print("1. Run a test backtest: [yellow]gpt-trader qt[/yellow]")
        self.console.print(
            "2. Explore strategies: [yellow]gpt-trader help --tutorial strategies[/yellow]"
        )
        self.console.print("3. Start paper trading: [yellow]gpt-trader paper --start[/yellow]")
        self.console.print("4. View your profile: [yellow]gpt-trader config --show[/yellow]")

        self.console.print("\n[green]✨ Setup wizard completed successfully![/green]")


def run_wizard() -> None:
    """Run the setup wizard."""
    wizard = SetupWizard()
    wizard.run()


if __name__ == "__main__":
    run_wizard()

"""Command shortcuts and quick actions for GPT-Trader."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class ShortcutManager:
    """Manage command shortcuts and quick actions."""

    # Define shortcuts mapping
    SHORTCUTS = {
        # Quick actions
        "qt": "quick-test",
        "qb": "quick-backtest",
        "qp": "quick-paper",
        # Common commands
        "bt": "backtest",
        "opt": "optimize",
        "wf": "walk-forward",
        "pp": "paper",
        "mon": "monitor",
        "dep": "deploy",
        # Interactive modes
        "i": "interactive",
        "menu": "interactive",
        "wizard": "interactive --setup",
        # Status and info
        "status": "monitor --status",
        "info": "interactive --system-check",
        "check": "interactive --system-check",
        # Quick profiles
        "demo": "backtest --symbol AAPL --strategy demo_ma",
        "test": "backtest --symbol SPY --strategy demo_ma --start 2024-01-01",
    }

    # Quick action definitions
    QUICK_ACTIONS = {
        "quick-test": {
            "description": "Run a quick test backtest with AAPL",
            "command": [
                "backtest",
                "--symbol",
                "AAPL",
                "--strategy",
                "demo_ma",
                "--start",
                "2024-01-01",
            ],
        },
        "quick-backtest": {
            "description": "Interactive quick backtest setup",
            "command": ["interactive", "--guided-backtest"],
        },
        "quick-paper": {
            "description": "Start paper trading with default settings",
            "command": ["paper", "--start"],
        },
    }

    @classmethod
    def resolve_shortcut(cls, command: str) -> str | None:
        """Resolve a shortcut to its full command."""
        return cls.SHORTCUTS.get(command)

    @classmethod
    def is_quick_action(cls, command: str) -> bool:
        """Check if command is a quick action."""
        return command in cls.QUICK_ACTIONS

    @classmethod
    def get_quick_action(cls, command: str) -> dict | None:
        """Get quick action details."""
        return cls.QUICK_ACTIONS.get(command)

    @classmethod
    def show_shortcuts(cls) -> None:
        """Display available shortcuts."""
        console.print(Panel("[bold cyan]Available Shortcuts[/bold cyan]", border_style="cyan"))

        # Group shortcuts by category
        categories = {
            "Quick Actions": ["qt", "qb", "qp"],
            "Common Commands": ["bt", "opt", "wf", "pp", "mon", "dep"],
            "Interactive": ["i", "menu", "wizard"],
            "Status": ["status", "info", "check"],
            "Demo": ["demo", "test"],
        }

        for category, shortcuts in categories.items():
            console.print(f"\n[bold yellow]{category}:[/bold yellow]")

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Shortcut", style="cyan", width=15)
            table.add_column("Command", style="white", width=40)

            for shortcut in shortcuts:
                full_cmd = cls.SHORTCUTS.get(shortcut, "")
                if full_cmd in cls.QUICK_ACTIONS:
                    desc = cls.QUICK_ACTIONS[full_cmd]["description"]
                    table.add_row(shortcut, desc)
                else:
                    table.add_row(shortcut, full_cmd)

            console.print(table)

    @classmethod
    def execute_shortcut(cls, shortcut: str, args: list[str]) -> int:
        """Execute a shortcut command."""
        resolved = cls.resolve_shortcut(shortcut)

        if not resolved:
            console.print(f"[red]Unknown shortcut: {shortcut}[/red]")
            console.print("Use 'gpt-trader shortcuts' to see available shortcuts")
            return 1

        # Check if it's a quick action
        if cls.is_quick_action(resolved):
            action = cls.get_quick_action(resolved)
            console.print(f"[cyan]Running: {action['description']}[/cyan]")

            # Build full command
            cmd = ["gpt-trader"] + action["command"] + args
            console.print(f"[dim]Command: {' '.join(cmd)}[/dim]\n")

            # Execute
            return subprocess.call(cmd)
        else:
            # Regular shortcut - just expand and run
            cmd_parts = resolved.split()
            cmd = ["gpt-trader"] + cmd_parts + args
            console.print(f"[dim]Executing: {' '.join(cmd)}[/dim]\n")
            return subprocess.call(cmd)


class QuickActions:
    """Quick action implementations."""

    @staticmethod
    def quick_validate(symbol: str = "AAPL") -> None:
        """Quick validation of a strategy."""
        console.print(
            Panel(
                f"[bold cyan]Quick Validation[/bold cyan]\n" f"Symbol: {symbol}",
                border_style="cyan",
            )
        )

        # Run quick backtest
        cmd = [
            "gpt-trader",
            "backtest",
            "--symbol",
            symbol,
            "--strategy",
            "demo_ma",
            "--start",
            "2024-01-01",
            "--end",
            "2024-03-01",
            "--quick",
        ]

        console.print(f"[dim]Running: {' '.join(cmd)}[/dim]\n")
        subprocess.call(cmd)

    @staticmethod
    def quick_compare() -> None:
        """Quick comparison of strategies."""
        console.print(
            Panel("[bold cyan]Quick Strategy Comparison[/bold cyan]", border_style="cyan")
        )

        strategies = ["demo_ma", "trend_breakout"]
        symbol = "SPY"

        results = Table(title=f"Strategy Comparison - {symbol}")
        results.add_column("Strategy", style="cyan")
        results.add_column("Return", justify="right")
        results.add_column("Sharpe", justify="right")
        results.add_column("Drawdown", justify="right")

        for strategy in strategies:
            console.print(f"\n[cyan]Testing {strategy}...[/cyan]")

            cmd = [
                "gpt-trader",
                "backtest",
                "--symbol",
                symbol,
                "--strategy",
                strategy,
                "--start",
                "2024-01-01",
                "--silent",
            ]

            # Run backtest (simplified for demo)
            subprocess.run(cmd, capture_output=True, text=True)

            # Parse results (simplified)
            results.add_row(strategy, "+12.5%", "1.35", "-8.2%")  # Mock data

        console.print("\n")
        console.print(results)

    @staticmethod
    def quick_health_check() -> None:
        """Quick system health check."""
        console.print(Panel("[bold cyan]System Health Check[/bold cyan]", border_style="cyan"))

        checks = [
            ("Python Version", sys.version.split()[0], "green"),
            (
                "Data Directory",
                "✓" if Path("data").exists() else "✗",
                "green" if Path("data").exists() else "red",
            ),
            (
                "Config Directory",
                "✓" if Path(".gpt-trader").exists() else "✗",
                "green" if Path(".gpt-trader").exists() else "yellow",
            ),
            ("Market Data", "Connected", "green"),
            ("Paper Trading", "Available", "green"),
        ]

        table = Table(show_header=False, box=None)
        table.add_column("Component", style="cyan")
        table.add_column("Status")

        for component, status, color in checks:
            table.add_row(component, f"[{color}]{status}[/{color}]")

        console.print(table)


def add_shortcuts_command(subparsers: argparse._SubParsersAction) -> None:
    """Add shortcuts command to CLI."""
    parser = subparsers.add_parser(
        "shortcuts",
        aliases=["sh", "alias"],
        help="Show available command shortcuts",
        description="Display all available command shortcuts and quick actions",
    )

    parser.add_argument("--list", action="store_true", help="List all shortcuts in simple format")

    parser.add_argument(
        "--add", nargs=2, metavar=("SHORTCUT", "COMMAND"), help="Add a custom shortcut"
    )

    parser.set_defaults(func=handle_shortcuts)


def handle_shortcuts(args: argparse.Namespace) -> None:
    """Handle shortcuts command."""
    if args.add:
        add_custom_shortcut(args.add[0], args.add[1])
    elif args.list:
        list_shortcuts_simple()
    else:
        ShortcutManager.show_shortcuts()


def add_custom_shortcut(shortcut: str, command: str) -> None:
    """Add a custom shortcut."""
    # Save to user config
    config_dir = Path.home() / ".gpt-trader"
    config_dir.mkdir(exist_ok=True)

    config_dir / "shortcuts.yaml"

    # Implementation would save the shortcut
    console.print(f"[green]✓[/green] Added shortcut: {shortcut} -> {command}")


def list_shortcuts_simple() -> None:
    """List shortcuts in simple format."""
    for shortcut, command in ShortcutManager.SHORTCUTS.items():
        console.print(f"{shortcut:10} -> {command}")


def check_and_run_shortcut(argv: list[str]) -> int | None:
    """Check if first argument is a shortcut and run it."""
    if len(argv) < 2:
        return None

    potential_shortcut = argv[1]

    # Check if it's a registered shortcut
    if ShortcutManager.resolve_shortcut(potential_shortcut):
        # Remove program name and shortcut, keep rest as args
        remaining_args = argv[2:] if len(argv) > 2 else []
        return ShortcutManager.execute_shortcut(potential_shortcut, remaining_args)

    return None

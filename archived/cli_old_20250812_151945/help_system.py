"""Enhanced help system with examples and tutorials."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

console = Console()


class HelpSystem:
    """Enhanced help system with examples and guidance."""

    # Command examples database
    EXAMPLES = {
        "backtest": [
            {
                "title": "Simple backtest with single symbol",
                "command": "gpt-trader backtest --symbol AAPL --strategy demo_ma",
                "description": "Run a basic moving average strategy on Apple stock",
            },
            {
                "title": "Backtest with date range",
                "command": "gpt-trader backtest --symbol SPY --strategy trend_breakout --start 2023-01-01 --end 2023-12-31",
                "description": "Test trend breakout strategy on SPY for 2023",
            },
            {
                "title": "Multi-symbol backtest",
                "command": "gpt-trader backtest --symbols AAPL,MSFT,GOOGL --strategy demo_ma --risk-pct 2",
                "description": "Test multiple stocks with 2% risk per trade",
            },
            {
                "title": "Backtest with CSV universe",
                "command": "gpt-trader backtest --symbol-list universe.csv --strategy trend_breakout",
                "description": "Use a CSV file containing list of symbols",
            },
        ],
        "optimize": [
            {
                "title": "Basic parameter optimization",
                "command": "gpt-trader optimize --symbol AAPL --strategy demo_ma",
                "description": "Find optimal parameters for moving average strategy",
            },
            {
                "title": "Grid search optimization",
                "command": "gpt-trader optimize --symbol SPY --method grid --param-ranges sma_fast:10-50:5",
                "description": "Grid search for optimal SMA periods",
            },
            {
                "title": "Evolutionary optimization",
                "command": "gpt-trader optimize --symbol QQQ --method evolutionary --generations 50",
                "description": "Use genetic algorithm for optimization",
            },
        ],
        "paper": [
            {
                "title": "Start paper trading",
                "command": "gpt-trader paper --start --strategy trend_breakout",
                "description": "Begin paper trading with trend breakout strategy",
            },
            {
                "title": "Check paper trading status",
                "command": "gpt-trader paper --status",
                "description": "View current positions and P&L",
            },
            {
                "title": "Stop paper trading",
                "command": "gpt-trader paper --stop",
                "description": "Stop paper trading and save results",
            },
        ],
        "monitor": [
            {
                "title": "Live performance monitoring",
                "command": "gpt-trader monitor --live",
                "description": "Real-time monitoring dashboard",
            },
            {
                "title": "Risk monitoring",
                "command": "gpt-trader monitor --risk",
                "description": "Monitor risk metrics and exposure",
            },
            {
                "title": "Generate performance report",
                "command": "gpt-trader monitor --report --output report.pdf",
                "description": "Create PDF performance report",
            },
        ],
    }

    # Tutorials
    TUTORIALS = {
        "getting_started": """
# Getting Started with GPT-Trader

## 1. Initial Setup
First, ensure you have GPT-Trader installed and configured:

```bash
# Check installation
gpt-trader --version

# Run system check
gpt-trader check

# Interactive setup
gpt-trader wizard
```

## 2. Your First Backtest
Start with a simple backtest to understand the system:

```bash
# Test Apple stock with moving average strategy
gpt-trader backtest --symbol AAPL --strategy demo_ma

# Or use the shortcut
gpt-trader bt --symbol AAPL
```

## 3. Understanding Results
After running a backtest, you'll see:
- Total return and Sharpe ratio
- Maximum drawdown
- Trade statistics
- Performance chart (if not disabled)

## 4. Next Steps
- Try different strategies: `trend_breakout`, `optimized_ma`
- Test multiple symbols
- Optimize parameters
- Start paper trading
        """,
        "strategies": """
# Available Strategies

## 1. Demo MA (Moving Average)
Simple moving average crossover strategy.

**Parameters:**
- `sma_fast`: Fast moving average period (default: 20)
- `sma_slow`: Slow moving average period (default: 50)

**Usage:**
```bash
gpt-trader backtest --symbol AAPL --strategy demo_ma
```

## 2. Trend Breakout
Donchian channel breakout with ATR-based stops.

**Parameters:**
- `donchian`: Channel lookback period (default: 55)
- `atr`: ATR period for stops (default: 20)
- `atr_k`: ATR multiplier (default: 2.0)

**Usage:**
```bash
gpt-trader backtest --symbol SPY --strategy trend_breakout \\
    --donchian 20 --atr 14 --atr-k 1.5
```

## 3. Optimized MA
Enhanced moving average with dynamic parameters.

**Features:**
- Adaptive periods based on volatility
- Volume confirmation
- Regime filtering

**Usage:**
```bash
gpt-trader backtest --symbol QQQ --strategy optimized_ma
```
        """,
        "risk_management": """
# Risk Management

## Position Sizing
Control risk per trade and overall exposure:

```bash
# 2% risk per trade, max 5 positions
gpt-trader backtest --risk-pct 2 --max-positions 5
```

## Stop Losses
Strategies implement different stop loss methods:

- **Fixed Stop**: Percentage-based stop loss
- **ATR Stop**: Volatility-adjusted stops
- **Trailing Stop**: Locks in profits

## Portfolio Constraints
Manage overall portfolio risk:

```bash
# Maximum 30% in any sector
gpt-trader backtest --max-sector-weight 30

# Minimum correlation between positions
gpt-trader backtest --min-correlation -0.3
```

## Risk Monitoring
Track risk metrics in real-time:

```bash
# Monitor risk dashboard
gpt-trader monitor --risk

# Set risk alerts
gpt-trader monitor --alert-drawdown 10
```
        """,
    }

    @classmethod
    def show_command_help(cls, command: str) -> None:
        """Show detailed help for a specific command."""
        examples = cls.EXAMPLES.get(command, [])

        if not examples:
            console.print(f"[yellow]No examples found for command: {command}[/yellow]")
            return

        console.print(
            Panel(f"[bold cyan]{command.upper()} Command Examples[/bold cyan]", border_style="cyan")
        )

        for i, example in enumerate(examples, 1):
            console.print(f"\n[bold yellow]{i}. {example['title']}[/bold yellow]")
            console.print(f"[dim]{example['description']}[/dim]\n")

            syntax = Syntax(example["command"], "bash", theme="monokai", line_numbers=False)
            console.print(syntax)

    @classmethod
    def show_tutorial(cls, topic: str) -> None:
        """Show a tutorial on a specific topic."""
        tutorial = cls.TUTORIALS.get(topic)

        if not tutorial:
            console.print(f"[yellow]No tutorial found for topic: {topic}[/yellow]")
            cls.list_tutorials()
            return

        md = Markdown(tutorial)
        console.print(Panel(md, border_style="cyan"))

    @classmethod
    def list_tutorials(cls) -> None:
        """List available tutorials."""
        console.print("\n[bold cyan]Available Tutorials:[/bold cyan]")

        tutorials = [
            ("getting_started", "Introduction and first steps"),
            ("strategies", "Available trading strategies"),
            ("risk_management", "Risk and money management"),
        ]

        table = Table(show_header=False, box=None)
        table.add_column("Topic", style="yellow")
        table.add_column("Description")

        for topic, desc in tutorials:
            table.add_row(topic, desc)

        console.print(table)
        console.print("\n[dim]Use: gpt-trader help --tutorial <topic>[/dim]")

    @classmethod
    def show_quick_reference(cls) -> None:
        """Show quick reference card."""
        console.print(
            Panel("[bold cyan]GPT-Trader Quick Reference[/bold cyan]", border_style="cyan")
        )

        # Commands tree
        tree = Tree("[bold]Commands")

        # Strategy Development
        strategy = tree.add("[yellow]Strategy Development")
        strategy.add("backtest - Test strategies historically")
        strategy.add("optimize - Find optimal parameters")
        strategy.add("walk-forward - Validate robustness")

        # Trading
        trading = tree.add("[yellow]Trading")
        trading.add("paper - Paper trading simulation")
        trading.add("deploy - Deploy to live trading")
        trading.add("monitor - Monitor performance")

        # Utilities
        utils = tree.add("[yellow]Utilities")
        utils.add("interactive - Interactive mode")
        utils.add("shortcuts - View shortcuts")
        utils.add("help - Get help")

        console.print(tree)

        # Common flags
        console.print("\n[bold yellow]Common Flags:[/bold yellow]")

        flags = Table(show_header=False, box=None)
        flags.add_column("Flag", style="cyan")
        flags.add_column("Description")

        flags.add_row("--symbol", "Single symbol to trade")
        flags.add_row("--symbols", "Comma-separated symbols")
        flags.add_row("--strategy", "Strategy to use")
        flags.add_row("--start/--end", "Date range")
        flags.add_row("--risk-pct", "Risk per trade")
        flags.add_row("--verbose", "Increase output")

        console.print(flags)

        # Shortcuts
        console.print("\n[bold yellow]Useful Shortcuts:[/bold yellow]")

        shortcuts = Table(show_header=False, box=None)
        shortcuts.add_column("Shortcut", style="cyan")
        shortcuts.add_column("Command")

        shortcuts.add_row("bt", "backtest")
        shortcuts.add_row("opt", "optimize")
        shortcuts.add_row("i", "interactive")
        shortcuts.add_row("qt", "quick test")

        console.print(shortcuts)

    @classmethod
    def show_faq(cls) -> None:
        """Show frequently asked questions."""
        console.print(
            Panel("[bold cyan]Frequently Asked Questions[/bold cyan]", border_style="cyan")
        )

        faqs = [
            {
                "q": "How do I start paper trading?",
                "a": "Use `gpt-trader paper --start` to begin paper trading with default settings, "
                "or `gpt-trader interactive` for guided setup.",
            },
            {
                "q": "What data sources are supported?",
                "a": "Currently supports Yahoo Finance for free data. Premium data sources like "
                "Alpaca, Interactive Brokers, and Polygon are also supported with API keys.",
            },
            {
                "q": "How do I optimize strategy parameters?",
                "a": "Use `gpt-trader optimize` with your strategy. You can choose between grid search, "
                "random search, or evolutionary optimization methods.",
            },
            {
                "q": "Can I create custom strategies?",
                "a": "Yes! Create a Python file inheriting from BaseStrategy class and place it in "
                "src/bot/strategy/. See docs/STRATEGIES.md for details.",
            },
            {
                "q": "How do I monitor live trading?",
                "a": "Use `gpt-trader monitor --live` for real-time dashboard, or "
                "`gpt-trader monitor --status` for current snapshot.",
            },
        ]

        for i, faq in enumerate(faqs, 1):
            console.print(f"\n[bold yellow]Q{i}: {faq['q']}[/bold yellow]")
            console.print(f"[white]A: {faq['a']}[/white]")

    @classmethod
    def search_help(cls, query: str) -> None:
        """Search help content for a query."""
        console.print(f"\n[cyan]Searching for: {query}[/cyan]\n")

        results = []

        # Search in examples
        for cmd, examples in cls.EXAMPLES.items():
            for example in examples:
                if (
                    query.lower() in example["title"].lower()
                    or query.lower() in example["description"].lower()
                    or query.lower() in example["command"].lower()
                ):
                    results.append(("Example", cmd, example["title"]))

        # Search in tutorials
        for topic, content in cls.TUTORIALS.items():
            if query.lower() in content.lower():
                results.append(("Tutorial", topic, f"Tutorial on {topic}"))

        if results:
            console.print(f"[green]Found {len(results)} results:[/green]\n")

            table = Table(show_header=True)
            table.add_column("Type", style="cyan")
            table.add_column("Location", style="yellow")
            table.add_column("Description")

            for type_, location, desc in results:
                table.add_row(type_, location, desc)

            console.print(table)
        else:
            console.print(f"[yellow]No results found for: {query}[/yellow]")


def add_help_command(subparsers: argparse._SubParsersAction) -> None:
    """Add enhanced help command to CLI."""
    parser = subparsers.add_parser(
        "help",
        help="Get detailed help and examples",
        description="Enhanced help system with examples and tutorials",
    )

    parser.add_argument("command", nargs="?", help="Command to get help for")

    parser.add_argument("--examples", action="store_true", help="Show command examples")

    parser.add_argument("--tutorial", metavar="TOPIC", help="Show tutorial on topic")

    parser.add_argument("--faq", action="store_true", help="Show frequently asked questions")

    parser.add_argument("--quick", action="store_true", help="Show quick reference card")

    parser.add_argument("--search", metavar="QUERY", help="Search help content")

    parser.set_defaults(func=handle_help)


def handle_help(args: argparse.Namespace) -> None:
    """Handle help command."""
    if args.tutorial:
        HelpSystem.show_tutorial(args.tutorial)
    elif args.faq:
        HelpSystem.show_faq()
    elif args.quick:
        HelpSystem.show_quick_reference()
    elif args.search:
        HelpSystem.search_help(args.search)
    elif args.command:
        HelpSystem.show_command_help(args.command)
    elif args.examples:
        # Show all examples
        for command in HelpSystem.EXAMPLES:
            HelpSystem.show_command_help(command)
            console.print("\n" + "=" * 50 + "\n")
    else:
        # Default: show quick reference
        HelpSystem.show_quick_reference()
        console.print("\n[dim]For more help: gpt-trader help --tutorial getting_started[/dim]")

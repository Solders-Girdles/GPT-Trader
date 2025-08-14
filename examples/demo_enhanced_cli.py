#!/usr/bin/env python
"""
Demo of the Enhanced CLI interface using Rich.
Shows the beautiful terminal UI without requiring full system imports.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import asyncio
import time

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TaskProgressColumn
)
from rich.prompt import Prompt, Confirm
from rich.tree import Tree
from rich.status import Status

# Initialize Rich console and Typer app
console = Console()
app = typer.Typer(
    name="gpt-trader",
    help="🚀 Professional Trading Platform with Beautiful CLI",
    add_completion=True,
    rich_markup_mode="rich"
)

# Color scheme
COLORS = {
    "profit": "green",
    "loss": "red",
    "warning": "yellow",
    "info": "cyan",
    "header": "bold magenta",
    "success": "bold green",
    "error": "bold red"
}


class TradingCLI:
    """Enhanced trading CLI with Rich interface."""
    
    def __init__(self):
        self.console = console
    
    def print_header(self):
        """Print beautiful header."""
        header = Panel.fit(
            "[bold cyan]GPT-Trader[/bold cyan] 🚀\n"
            "[dim]Professional Algorithmic Trading Platform[/dim]",
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(header)
    
    def create_portfolio_table(self, positions: Dict[str, Any]) -> Table:
        """Create a beautiful portfolio table."""
        table = Table(
            title="📊 Portfolio Positions",
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            show_lines=True,
            title_style="bold cyan"
        )
        
        table.add_column("Symbol", style="cyan", width=10)
        table.add_column("Quantity", justify="right", width=10)
        table.add_column("Entry Price", justify="right", width=12)
        table.add_column("Current Price", justify="right", width=12)
        table.add_column("Market Value", justify="right", width=15)
        table.add_column("P&L", justify="right", width=15)
        table.add_column("P&L %", justify="right", width=10)
        
        total_value = 0
        total_pnl = 0
        
        for symbol, pos in positions.items():
            qty = pos.get("quantity", 0)
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", entry)
            value = qty * current
            pnl = qty * (current - entry)
            pnl_pct = ((current / entry) - 1) * 100 if entry > 0 else 0
            
            total_value += value
            total_pnl += pnl
            
            # Color code P&L
            pnl_color = COLORS["profit"] if pnl >= 0 else COLORS["loss"]
            
            table.add_row(
                symbol,
                str(qty),
                f"${entry:.2f}",
                f"${current:.2f}",
                f"${value:,.2f}",
                f"[{pnl_color}]${pnl:+,.2f}[/{pnl_color}]",
                f"[{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]"
            )
        
        # Add totals row
        table.add_section()
        total_pnl_color = COLORS["profit"] if total_pnl >= 0 else COLORS["loss"]
        table.add_row(
            "[bold]TOTAL[/bold]",
            "",
            "",
            "",
            f"[bold]${total_value:,.2f}[/bold]",
            f"[bold {total_pnl_color}]${total_pnl:+,.2f}[/bold {total_pnl_color}]",
            "",
            style="bold"
        )
        
        return table
    
    def create_performance_panel(self, metrics: Dict[str, float]) -> Panel:
        """Create performance metrics panel."""
        sharpe = metrics.get("sharpe_ratio", 0)
        returns = metrics.get("total_return", 0) * 100
        max_dd = metrics.get("max_drawdown", 0) * 100
        win_rate = metrics.get("win_rate", 0) * 100
        
        # Color code metrics
        returns_color = COLORS["profit"] if returns >= 0 else COLORS["loss"]
        sharpe_color = COLORS["profit"] if sharpe >= 1 else COLORS["warning"] if sharpe >= 0 else COLORS["loss"]
        
        content = f"""
[bold]Performance Metrics[/bold]
═══════════════════════════════

📈 [bold]Total Return:[/bold] [{returns_color}]{returns:+.2f}%[/{returns_color}]
📊 [bold]Sharpe Ratio:[/bold] [{sharpe_color}]{sharpe:.2f}[/{sharpe_color}]
📉 [bold]Max Drawdown:[/bold] [red]{max_dd:.2f}%[/red]
🎯 [bold]Win Rate:[/bold] {win_rate:.1f}%
💰 [bold]Total Trades:[/bold] {metrics.get('total_trades', 0)}
⏱️ [bold]Avg Hold Time:[/bold] {metrics.get('avg_hold_days', 0):.1f} days
        """
        
        return Panel(
            content.strip(),
            title="📊 Performance Summary",
            border_style="green" if returns >= 0 else "red",
            padding=(1, 2)
        )
    
    def show_strategy_tree(self, strategies: List[str]):
        """Display available strategies in a tree structure."""
        tree = Tree("📋 [bold cyan]Available Strategies[/bold cyan]")
        
        categories = {
            "Trend Following": ["trend_breakout", "demo_ma"],
            "Mean Reversion": ["bollinger_bands", "rsi_oversold"],
            "Momentum": ["momentum_breakout", "relative_strength"],
            "Machine Learning": ["random_forest", "neural_network"]
        }
        
        for category, strats in categories.items():
            branch = tree.add(f"[bold yellow]{category}[/bold yellow]")
            for strat in strats:
                if strat in strategies:
                    branch.add(f"✅ [green]{strat}[/green]")
                else:
                    branch.add(f"⭕ [dim]{strat}[/dim] [dim italic](not available)[/dim italic]")
        
        self.console.print(tree)
    
    def get_sample_positions(self) -> Dict[str, Any]:
        """Get sample portfolio positions."""
        return {
            "AAPL": {"quantity": 100, "entry_price": 150.00, "current_price": 155.50},
            "GOOGL": {"quantity": 10, "entry_price": 2800.00, "current_price": 2850.00},
            "MSFT": {"quantity": 50, "entry_price": 300.00, "current_price": 295.00},
            "TSLA": {"quantity": 25, "entry_price": 200.00, "current_price": 215.00},
            "AMZN": {"quantity": 20, "entry_price": 140.00, "current_price": 142.50}
        }
    
    def get_sample_metrics(self) -> Dict[str, float]:
        """Get sample performance metrics."""
        return {
            "total_return": 0.0523,
            "sharpe_ratio": 1.45,
            "max_drawdown": -0.0820,
            "win_rate": 0.58,
            "total_trades": 42,
            "avg_hold_days": 3.2
        }


# CLI instance
cli = TradingCLI()


@app.command()
def demo():
    """
    🎬 Run a complete demo of the enhanced CLI features.
    """
    cli.print_header()
    console.print()
    
    # Show portfolio
    console.print("[bold cyan]📊 Current Portfolio Status[/bold cyan]\n")
    positions = cli.get_sample_positions()
    console.print(cli.create_portfolio_table(positions))
    console.print()
    
    # Show performance
    metrics = cli.get_sample_metrics()
    console.print(cli.create_performance_panel(metrics))
    console.print()
    
    # Show available strategies
    console.print("[bold cyan]📋 Trading Strategies[/bold cyan]\n")
    available = ["trend_breakout", "demo_ma", "momentum_breakout"]
    cli.show_strategy_tree(available)


@app.command()
def backtest(
    strategy: str = typer.Argument(..., help="Strategy name to backtest"),
    symbols: str = typer.Option("AAPL,GOOGL,MSFT", help="Comma-separated symbols"),
    start: str = typer.Option("2023-01-01", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2023-12-31", help="End date (YYYY-MM-DD)"),
    capital: float = typer.Option(100000.0, help="Starting capital")
):
    """
    🎯 Simulate a backtest with beautiful progress tracking.
    """
    cli.print_header()
    
    # Parse symbols
    symbol_list = [s.strip() for s in symbols.split(",")]
    
    console.print(f"\n[bold cyan]Starting Backtest[/bold cyan]")
    console.print(f"Strategy: [yellow]{strategy}[/yellow]")
    console.print(f"Symbols: [yellow]{', '.join(symbol_list)}[/yellow]")
    console.print(f"Period: [yellow]{start}[/yellow] to [yellow]{end}[/yellow]")
    console.print(f"Capital: [yellow]${capital:,.2f}[/yellow]\n")
    
    # Create progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        # Task 1: Load data
        task1 = progress.add_task("[cyan]Loading market data...", total=len(symbol_list))
        
        for symbol in symbol_list:
            time.sleep(0.5)  # Simulate loading
            progress.update(task1, advance=1, description=f"[cyan]Loading {symbol}...")
        
        # Task 2: Run backtest
        task2 = progress.add_task("[yellow]Running backtest...", total=100)
        
        for i in range(100):
            time.sleep(0.02)  # Simulate processing
            progress.update(task2, advance=1)
    
    # Display results
    console.print("\n")
    
    # Portfolio table
    positions = cli.get_sample_positions()
    console.print(cli.create_portfolio_table(positions))
    
    console.print("\n")
    
    # Performance panel
    metrics = cli.get_sample_metrics()
    console.print(cli.create_performance_panel(metrics))
    
    # Success message
    console.print(
        Panel(
            f"[bold green]✅ Backtest completed successfully![/bold green]\n"
            f"Strategy: [cyan]{strategy}[/cyan]\n"
            f"Period: {start} to {end}\n"
            f"Initial Capital: ${capital:,.2f}",
            border_style="green"
        )
    )


@app.command()
def strategies():
    """
    📋 List all available trading strategies.
    """
    cli.print_header()
    console.print()
    
    available = ["trend_breakout", "demo_ma", "momentum_breakout"]
    cli.show_strategy_tree(available)


@app.command()
def risk():
    """
    ⚠️ Display current risk metrics and limits.
    """
    cli.print_header()
    console.print()
    
    # Create risk table
    table = Table(
        title="⚠️ Risk Management Status",
        show_header=True,
        header_style="bold red",
        border_style="yellow",
        title_style="bold yellow"
    )
    
    table.add_column("Metric", style="cyan")
    table.add_column("Current", justify="right")
    table.add_column("Limit", justify="right")
    table.add_column("Status", justify="center")
    
    risk_metrics = [
        ("Portfolio VaR (95%)", "$2,450", "$5,000", "✅"),
        ("Max Drawdown", "-8.2%", "-15%", "✅"),
        ("Portfolio Beta", "0.95", "1.20", "✅"),
        ("Leverage", "1.0x", "2.0x", "✅"),
        ("Concentration Risk", "28%", "30%", "⚠️"),
        ("Daily Loss", "-1.2%", "-3%", "✅")
    ]
    
    for metric, current, limit, status in risk_metrics:
        table.add_row(metric, current, limit, status)
    
    console.print(table)
    console.print()
    
    # Risk summary
    console.print(
        Panel(
            "[bold green]Risk Status: HEALTHY[/bold green]\n"
            "All risk metrics within acceptable limits.\n"
            "[dim]Last updated: " + datetime.now().strftime("%H:%M:%S") + "[/dim]",
            border_style="green",
            padding=(1, 2)
        )
    )


@app.command()
def portfolio():
    """
    💼 Display current portfolio positions.
    """
    cli.print_header()
    console.print()
    
    positions = cli.get_sample_positions()
    console.print(cli.create_portfolio_table(positions))


@app.command()
def performance():
    """
    📈 Show performance metrics.
    """
    cli.print_header()
    console.print()
    
    metrics = cli.get_sample_metrics()
    console.print(cli.create_performance_panel(metrics))


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version")
):
    """
    🚀 GPT-Trader: Professional Algorithmic Trading Platform
    
    Beautiful CLI for backtesting, live trading, and portfolio management.
    """
    if version:
        console.print("[bold cyan]GPT-Trader v2.0.0[/bold cyan]")
        console.print("[dim]Enhanced with Rich UI[/dim]")
        raise typer.Exit()


if __name__ == "__main__":
    app()
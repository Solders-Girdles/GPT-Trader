#!/usr/bin/env python3
"""
Paper Trading Runner Script.

Runs the trading bot in paper trading mode with real Coinbase market data
and simulated order execution. Uses 'rich' for enhanced CLI experience.

Usage:
    python scripts/run_paper_trading.py
    python scripts/run_paper_trading.py --single-cycle
    python scripts/run_paper_trading.py --interval 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpt_trader.app.container import create_application_container
from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.features.brokerages.paper import HybridPaperBroker
from gpt_trader.utilities.logging_patterns import get_logger

# Disable standard logging to avoid cluttering rich output
import logging

logging.getLogger("gpt_trader").setLevel(logging.WARNING)

console = Console()


class PaperTradingEngine:
    """Simple paper trading engine using HybridPaperBroker."""

    def __init__(
        self,
        broker: HybridPaperBroker,
        symbols: list[str],
        interval: int = 60,
    ) -> None:
        self.broker = broker
        self.symbols = symbols
        self.interval = interval
        self.running = True
        self.cycle_count = 0

        # Simple MA crossover state
        self._price_history: dict[str, list[Decimal]] = {s: [] for s in symbols}
        self._short_window = 5
        self._long_window = 20

    def _calculate_sma(self, prices: list[Decimal], window: int) -> Decimal | None:
        """Calculate simple moving average."""
        if len(prices) < window:
            return None
        return sum(prices[-window:]) / window

    def _generate_signal(self, symbol: str, price: Decimal) -> str | None:
        """Generate trading signal based on MA crossover."""
        self._price_history[symbol].append(price)

        # Keep only needed history
        max_history = self._long_window + 1
        if len(self._price_history[symbol]) > max_history:
            self._price_history[symbol] = self._price_history[symbol][-max_history:]

        short_ma = self._calculate_sma(self._price_history[symbol], self._short_window)
        long_ma = self._calculate_sma(self._price_history[symbol], self._long_window)

        if short_ma is None or long_ma is None:
            return None

        # Simple crossover logic
        if short_ma > long_ma * Decimal("1.001"):  # Short above long by 0.1%
            return "buy"
        elif short_ma < long_ma * Decimal("0.999"):  # Short below long by 0.1%
            return "sell"
        return None

    async def run_cycle(self) -> None:
        """Run a single trading cycle."""
        self.cycle_count += 1
        cycle_start = time.time()

        console.print(
            f"[bold blue]Cycle {self.cycle_count}[/bold blue] starting at {datetime.now().strftime('%H:%M:%S')}"
        )

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Symbol")
        table.add_column("Price", justify="right")
        table.add_column("Signal", justify="center")
        table.add_column("Action", justify="right")

        for symbol in self.symbols:
            try:
                # Get current quote
                quote = self.broker.get_quote(symbol)
                if not quote:
                    console.print(f"[yellow]No quote for {symbol}[/yellow]")
                    continue

                price = quote.last
                signal_type = self._generate_signal(symbol, price)
                action_msg = "-"

                # Check current position
                positions = {p.symbol: p for p in self.broker.list_positions()}
                current_pos = positions.get(symbol)

                if signal_type == "buy" and not current_pos:
                    # Calculate position size (1% of equity)
                    equity = self.broker.get_equity()
                    position_value = equity * Decimal("0.01")
                    quantity = position_value / price

                    if quantity >= Decimal("0.0001"):
                        self.broker.place_order(
                            symbol_or_payload=symbol,
                            side="buy",
                            order_type="market",
                            quantity=quantity.quantize(Decimal("0.0001")),
                        )
                        action_msg = f"[green]BUY {quantity:.4f}[/green]"

                elif signal_type == "sell" and current_pos and current_pos.quantity > 0:
                    self.broker.place_order(
                        symbol_or_payload=symbol,
                        side="sell",
                        order_type="market",
                        quantity=current_pos.quantity,
                    )
                    action_msg = f"[red]SELL {current_pos.quantity:.4f}[/red]"

                sig_display = f"[bold]{signal_type.upper()}[/bold]" if signal_type else "-"
                if signal_type == "buy":
                    sig_display = f"[green]{sig_display}[/green]"
                if signal_type == "sell":
                    sig_display = f"[red]{sig_display}[/red]"

                table.add_row(symbol, f"${price:,.2f}", sig_display, action_msg)

            except Exception as e:
                console.print(f"[bold red]Error processing {symbol}: {e}[/bold red]")

        console.print(table)

        # Log status
        cycle_time = time.time() - cycle_start
        status = self.broker.get_status()

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Equity", f"${status['current_equity']:,.2f}")
        summary.add_row("Positions", str(status["positions"]))
        summary.add_row("Orders", str(status["orders_executed"]))
        summary.add_row("Cycle Time", f"{cycle_time:.2f}s")

        console.print(Panel(summary, title="Session Status", border_style="cyan"))

    async def run(self, single_cycle: bool = False) -> None:
        """Run the trading loop."""
        console.print(
            Panel.fit(
                f"[bold]Paper Trading Engine[/bold]\n"
                f"Symbols: {self.symbols}\n"
                f"Interval: {self.interval}s\n"
                f"Initial Equity: ${self.broker.get_equity():,.2f}",
                style="bold white on blue",
            )
        )

        try:
            while self.running:
                await self.run_cycle()

                if single_cycle:
                    console.print("[yellow]Single cycle complete, exiting.[/yellow]")
                    break

                with console.status(
                    f"[dim]Waiting {self.interval}s until next cycle...[/dim]", spinner="dots"
                ):
                    await asyncio.sleep(self.interval)

        except asyncio.CancelledError:
            console.print("[yellow]Trading loop cancelled.[/yellow]")

        # Final status
        final_status = self.broker.get_status()
        pnl = final_status["current_equity"] - final_status["initial_equity"]
        pnl_pct = (pnl / final_status["initial_equity"]) * 100
        pnl_color = "green" if pnl >= 0 else "red"

        console.print("\n")
        console.print(
            Panel(
                f"Total Cycles: {self.cycle_count}\n"
                f"Initial Equity: ${final_status['initial_equity']:,.2f}\n"
                f"Final Equity:   ${final_status['current_equity']:,.2f}\n"
                f"Net P&L:        [{pnl_color}]${pnl:,.2f} ({pnl_pct:+.2f}%)[/{pnl_color}]",
                title="PAPER TRADING SESSION COMPLETE",
                border_style="green",
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper Trading Runner")
    parser.add_argument("--single-cycle", action="store_true", help="Run single cycle and exit")
    parser.add_argument(
        "--interval", type=int, default=60, help="Interval between cycles in seconds"
    )
    parser.add_argument(
        "--symbols", type=str, default="BTC-USD", help="Comma-separated list of symbols"
    )
    parser.add_argument("--equity", type=float, default=10000, help="Initial paper trading equity")
    parser.add_argument(
        "--config", type=str, default="config/profiles/paper.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    try:
        with console.status("[bold green]Initializing...[/bold green]"):
            # Load config and create container
            config_path = Path(args.config)
            if config_path.exists():
                config = BotConfig.from_yaml(config_path)
            else:
                console.print(
                    f"[yellow]Config file {config_path} not found, using defaults[/yellow]"
                )
                config = BotConfig()

            # Override config with CLI args if necessary
            if args.symbols != "BTC-USD":
                config.symbols = [s.strip() for s in args.symbols.split(",")]

            container = create_application_container(config)

            # Create HybridPaperBroker using container's broker (CoinbaseClient)
            broker = HybridPaperBroker(
                client=container.broker,
                initial_equity=Decimal(str(args.equity)),
                slippage_bps=5,
                commission_bps=Decimal("5"),
            )

            symbols = config.symbols
            engine = PaperTradingEngine(broker=broker, symbols=symbols, interval=args.interval)

        # Signal handlers
        def signal_handler(sig: int, frame: Any) -> None:
            console.print(f"\n[yellow]Signal {sig} received, stopping...[/yellow]")
            engine.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        asyncio.run(engine.run(single_cycle=args.single_cycle))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Fatal Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

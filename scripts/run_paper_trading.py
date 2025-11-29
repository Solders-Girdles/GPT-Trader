#!/usr/bin/env python3
"""
Paper Trading Runner Script.

Runs the trading bot in paper trading mode with real Coinbase market data
and simulated order execution. This is a standalone script that bypasses
the full DI container for quick testing.

Usage:
    python scripts/run_paper_trading.py
    python scripts/run_paper_trading.py --single-cycle  # Run once and exit
    python scripts/run_paper_trading.py --interval 30   # 30 second cycles
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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpt_trader.orchestration.hybrid_paper_broker import HybridPaperBroker
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="paper_trading")


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

        logger.info(
            f"Cycle {self.cycle_count} starting at {datetime.now().isoformat()}",
            cycle=self.cycle_count,
        )

        for symbol in self.symbols:
            try:
                # Get current quote
                quote = self.broker.get_quote(symbol)
                if not quote:
                    logger.warning(f"No quote for {symbol}")
                    continue

                price = quote.last
                logger.info(f"{symbol}: ${price:.2f} (bid=${quote.bid:.2f}, ask=${quote.ask:.2f})")

                # Generate signal
                signal = self._generate_signal(symbol, price)

                # Check current position
                positions = {p.symbol: p for p in self.broker.list_positions()}
                current_pos = positions.get(symbol)

                if signal == "buy" and not current_pos:
                    # Calculate position size (1% of equity)
                    equity = self.broker.get_equity()
                    position_value = equity * Decimal("0.01")
                    quantity = position_value / price

                    if quantity >= Decimal("0.0001"):
                        logger.info(f"Signal: BUY {symbol}")
                        self.broker.place_order(
                            symbol_or_payload=symbol,
                            side="buy",
                            order_type="market",
                            quantity=quantity.quantize(Decimal("0.0001")),
                        )

                elif signal == "sell" and current_pos and current_pos.quantity > 0:
                    logger.info(f"Signal: SELL {symbol}")
                    self.broker.place_order(
                        symbol_or_payload=symbol,
                        side="sell",
                        order_type="market",
                        quantity=current_pos.quantity,
                    )

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Log status
        cycle_time = time.time() - cycle_start
        status = self.broker.get_status()
        logger.info(
            f"Cycle {self.cycle_count} complete in {cycle_time:.2f}s | "
            f"Equity: ${status['current_equity']:.2f} | "
            f"Positions: {status['positions']} | "
            f"Orders: {status['orders_executed']}"
        )

    async def run(self, single_cycle: bool = False) -> None:
        """Run the trading loop."""
        logger.info("Paper Trading Engine starting...")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Interval: {self.interval}s")
        logger.info(f"Initial equity: ${self.broker.get_equity():.2f}")

        try:
            while self.running:
                await self.run_cycle()

                if single_cycle:
                    logger.info("Single cycle complete, exiting.")
                    break

                logger.info(f"Sleeping {self.interval}s until next cycle...")
                await asyncio.sleep(self.interval)

        except asyncio.CancelledError:
            logger.info("Trading loop cancelled.")

        # Final status
        final_status = self.broker.get_status()
        logger.info("=" * 60)
        logger.info("PAPER TRADING SESSION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total cycles: {self.cycle_count}")
        logger.info(f"Initial equity: ${final_status['initial_equity']:.2f}")
        logger.info(f"Final equity: ${final_status['current_equity']:.2f}")
        pnl = final_status["current_equity"] - final_status["initial_equity"]
        logger.info(f"Net P&L: ${pnl:.2f} ({pnl / final_status['initial_equity'] * 100:.2f}%)")


def load_cdp_key(path: str | None = None) -> tuple[str, str]:
    """Load CDP API key from secrets file."""
    if path is None:
        project_root = Path(__file__).parent.parent
        path = str(project_root / "secrets" / "November2025APIKey.json")
    with open(path) as f:
        data = json.load(f)
    return data["name"], data["privateKey"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper Trading Runner")
    parser.add_argument(
        "--single-cycle",
        action="store_true",
        help="Run single cycle and exit",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Interval between cycles in seconds (default: 60)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC-USD",
        help="Comma-separated list of symbols (default: BTC-USD)",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=10000,
        help="Initial paper trading equity (default: 10000)",
    )
    args = parser.parse_args()

    # Load credentials
    logger.info("Loading CDP credentials...")
    api_key, private_key = load_cdp_key()

    # Create broker
    logger.info("Creating HybridPaperBroker...")
    broker = HybridPaperBroker(
        api_key=api_key,
        private_key=private_key,
        initial_equity=Decimal(str(args.equity)),
        slippage_bps=5,
        commission_bps=Decimal("5"),
    )

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]

    # Create engine
    engine = PaperTradingEngine(
        broker=broker,
        symbols=symbols,
        interval=args.interval,
    )

    # Signal handlers
    def signal_handler(sig: int, frame: Any) -> None:
        logger.info(f"Signal {sig} received, stopping...")
        engine.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    try:
        asyncio.run(engine.run(single_cycle=args.single_cycle))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

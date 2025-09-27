#!/usr/bin/env python3
"""
Unified Paper Trading Entry Point

Runs the Bot V2 orchestrator in paper mode with a simple CLI.

Examples:
  python scripts/paper_trade.py --symbols BTC-USD,ETH-USD --capital 10000 --cycles 5 --interval 15
  python scripts/paper_trade.py --symbols BTC-USD --sandbox --once

Notes:
  - Respects Coinbase environment variables for broker selection and sandbox.
  - Uses the existing orchestrator path and paper trade slice.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run paper trading via Bot V2 orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--symbols",
        default="BTC-USD",
        help="Comma-separated list of symbols (e.g., BTC-USD,ETH-USD)",
    )
    p.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        help="Starting capital in USD",
    )
    p.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of trading cycles to run (per symbol)",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between cycles",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit (overrides --cycles)",
    )
    p.add_argument(
        "--broker",
        default="coinbase",
        help="Broker to use (via env BROKER)",
    )
    p.add_argument(
        "--sandbox",
        action="store_true",
        help="Enable Coinbase sandbox mode (sets COINBASE_SANDBOX=1)",
    )
    p.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable console dashboard for monitoring",
    )
    p.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML summary report at the end",
    )
    return p.parse_args()


def main() -> int:
    _ensure_src_on_path()

    # Late imports after adjusting sys.path
    from bot_v2.orchestration import TradingOrchestrator, OrchestratorConfig, TradingMode
    from bot_v2.orchestration.execution import PaperExecutionEngine
    from bot_v2.features.paper_trade.dashboard import PaperTradingDashboard

    args = parse_args()

    # Normalize symbols
    symbols: List[str] = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("No symbols provided")
        return 1

    # Env wiring for broker selection and sandbox
    os.environ.setdefault("BROKER", args.broker)
    if args.sandbox:
        os.environ["COINBASE_SANDBOX"] = "1"

    # Build config and orchestrator
    config = OrchestratorConfig(
        mode=TradingMode.PAPER,
        symbols=symbols,
        capital=args.capital,
    )
    orchestrator = TradingOrchestrator(config)

    cycles = 1 if args.once else max(1, args.cycles)
    interval = max(0.0, args.interval)

    # Setup dashboard if requested
    # Create a paper execution engine for dashboard tracking
    dashboard = None
    paper_engine = None
    if args.dashboard:
        # Create a paper engine for tracking
        paper_engine = PaperExecutionEngine(
            initial_capital=args.capital,
            bot_id=f"paper:{''.join([s.replace('-', '') for s in symbols])}",
            symbols=symbols
        )
        dashboard = PaperTradingDashboard(paper_engine, refresh_interval=int(interval))

    print("🚀 Paper trading start")
    print(f"Symbols: {', '.join(symbols)} | Capital: ${args.capital:,.2f} | Cycles: {cycles}")
    if args.dashboard:
        print("📊 Dashboard enabled - displaying after each cycle")

    try:
        for i in range(cycles):
            for sym in symbols:
                res = orchestrator.execute_trading_cycle(sym)
                status = "✅" if res.success else "❌"
                eq = res.metrics.get("equity") or res.metrics.get("portfolio_equity")
                eq_str = f" | Equity: ${eq:,.2f}" if eq is not None else ""
                print(f"[{status}] {sym} | Mode={res.mode.value}{eq_str}")
                if res.errors:
                    print("  Errors:")
                    for e in res.errors[:3]:
                        print(f"   - {e}")
                
                # Update paper engine with the trading result if dashboard is enabled
                if paper_engine and res.data:
                    # Track any trades that may have happened
                    if 'paper_trade_result' in res.data:
                        trade_result = res.data['paper_trade_result']
                        if trade_result.get('action') == 'buy':
                            paper_engine.buy(sym, trade_result.get('quantity', 1000), 'orchestrator trade')
                        elif trade_result.get('action') == 'sell':
                            paper_engine.sell(sym, reason='orchestrator trade')
            
            # Display dashboard after each cycle if enabled
            if dashboard:
                print("\n" + "="*80)
                dashboard.display_once()
            
            if i < cycles - 1 and interval > 0:
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1

    # Generate HTML report if requested
    if args.html_report and dashboard:
        report_path = dashboard.generate_html_summary()
        print(f"\n📄 HTML report saved: {report_path}")

    print("🏁 Paper trading complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


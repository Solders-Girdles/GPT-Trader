from __future__ import annotations

import argparse
import sys
from datetime import datetime

from bot.backtest.engine import run_backtest as run_backtest_single
from bot.backtest.engine_portfolio import run_backtest as run_backtest_port
from bot.portfolio.allocator import PortfolioRules
from bot.strategy.base import Strategy
from bot.strategy.demo_ma import DemoMAStrategy
from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(prog="bot", description="GPT-Trader CLI")
    sub = parser.add_subparsers(dest="command")

    # Backtest subcommand
    p = sub.add_parser("backtest", help="Run a backtest")
    p.add_argument(
        "--strategy",
        default="trend_breakout",
        help="Strategy name (demo_ma, trend_breakout)",
    )
    p.add_argument("--symbol", help="Single symbol to backtest")
    p.add_argument("--symbol-list", help="CSV with 'symbol' or 'ticker' column")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")

    # Strategy params (for trend_breakout)
    p.add_argument("--donchian", type=int, default=55, help="Donchian lookback")
    p.add_argument("--atr", type=int, default=20, help="ATR period")
    p.add_argument("--atr-k", type=float, default=2.0, help="ATR stop multiplier")

    # Portfolio / risk params
    p.add_argument(
        "--risk-pct",
        type=float,
        default=0.5,
        help="Per-trade risk as percent of equity",
    )
    p.add_argument("--max-positions", type=int, default=10, help="Max concurrent positions")
    p.add_argument("--cost-bps", type=float, default=0.0, help="One-way transaction cost in bps")

    # Regime filter
    p.add_argument(
        "--regime",
        choices=["on", "off"],
        default="off",
        help="SPY > 200DMA regime filter",
    )
    p.add_argument("--regime-symbol", default="SPY", help="Regime symbol")
    p.add_argument("--regime-window", type=int, default=200, help="Regime SMA window")

    p.set_defaults(func=_handle_backtest)

    # Paper/live placeholders
    sub.add_parser("paper", help="Run paper trading (stub)")
    sub.add_parser("live", help="Run live trading (stub)")

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


def _handle_backtest(args: argparse.Namespace) -> None:
    # Build strategy
    strategy: Strategy
    if args.strategy == "demo_ma":
        strategy = DemoMAStrategy()
    elif args.strategy == "trend_breakout":
        strategy = TrendBreakoutStrategy(
            TrendBreakoutParams(
                donchian_lookback=args.donchian,
                atr_period=args.atr,
                atr_k=args.atr_k,
            )
        )
    else:
        print("Unknown strategy. Options: demo_ma, trend_breakout")
        sys.exit(1)

    # Build portfolio rules
    rules = PortfolioRules(
        per_trade_risk_pct=args.risk_pct / 100.0,
        atr_k=args.atr_k,
        max_positions=args.max_positions,
        cost_bps=args.cost_bps,
    )

    # Route: single vs portfolio engine
    if args.symbol and not args.symbol_list and args.strategy == "demo_ma":
        # simple single-symbol demo uses the single engine
        run_backtest_single(
            symbol=args.symbol,
            start=_parse_date(args.start),
            end=_parse_date(args.end),
            strategy=strategy,
        )
    else:
        # portfolio engine (supports both single & list; regime/costs honored)
        run_backtest_port(
            symbol=args.symbol,
            symbol_list_csv=args.symbol_list,
            start=_parse_date(args.start),
            end=_parse_date(args.end),
            strategy=strategy,
            rules=rules,
            regime_on=(args.regime == "on"),
            regime_symbol=args.regime_symbol,
            regime_window=args.regime_window,
        )


if __name__ == "__main__":
    main()

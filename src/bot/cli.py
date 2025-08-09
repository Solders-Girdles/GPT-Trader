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
    p.add_argument("--debug", action="store_true", help="Write per-day debug CSVs")
    p.add_argument(
        "--entry-confirm",
        type=int,
        default=1,
        help="Require N consecutive buy signals before a new entry (1=off)",
    )
    p.add_argument(
        "--min-rebalance-pct",
        type=float,
        default=0.0,
        help="Skip trades if |Δnotional| < pct of equity (e.g., 0.002 = 0.2%)",
    )
    p.add_argument(
        "--cadence",
        choices=["daily", "weekly"],
        default="daily",
        help="Rebalance frequency (daily or Mondays only)",
    )
    p.add_argument(
        "--cooldown",
        type=int,
        default=0,
        help="Bars to wait after an exit before re-entry (0 = off)",
    )
    p.add_argument(
        "--exit-mode",
        choices=["signal", "stop"],
        default="signal",
        help="Exit on signal/regime (signal) or hold until ATR stop (stop)",
    )

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
    strategy: Strategy  # This tells mypy "this can be any Strategy subclass"

    # Build strategy
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
        raise ValueError(f"Unknown strategy: {args.strategy}")

    # Common portfolio rules
    rules = PortfolioRules(
        per_trade_risk_pct=args.risk_pct / 100.0,
        atr_k=args.atr_k,
        max_positions=args.max_positions,
        cost_bps=args.cost_bps,
    )

    # Choose engine based on whether a list was supplied
    if args.symbol and not args.symbol_list:
        # single-symbol simple engine (no debug flag here)
        run_backtest_single(
            symbol=args.symbol,
            start=_parse_date(args.start),
            end=_parse_date(args.end),
            strategy=strategy,
        )
    else:
        # portfolio engine (supports debug + regime filter)
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
            debug=args.debug,
            exit_mode=args.exit_mode,
            cadence=args.cadence,
            cooldown=args.cooldown,
            entry_confirm=args.entry_confirm,
            min_rebalance_pct=args.min_rebalance_pct,
        )


if __name__ == "__main__":
    main()

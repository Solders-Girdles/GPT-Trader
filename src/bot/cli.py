from __future__ import annotations

import argparse
import sys
from datetime import datetime

from .backtest.engine import run_backtest
from .logging import get_logger
from .strategy.demo_ma import DemoMAStrategy

logger = get_logger("cli")


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(prog="bot", description="GPT-Trader CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Backtest
    p_back = sub.add_parser("backtest", help="Run a backtest")
    p_back.add_argument("--strategy", default="demo_ma", help="Strategy name (demo_ma)")
    p_back.add_argument("--symbol", required=True, help="Ticker symbol (e.g., AAPL)")
    p_back.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p_back.add_argument("--end", required=True, help="End date YYYY-MM-DD")

    # Paper/live (stubs for now)
    p_paper = sub.add_parser("paper", help="Run paper trading (stub)")
    p_paper.add_argument("--strategy", default="demo_ma")

    p_live = sub.add_parser("live", help="Run live trading (stub)")
    p_live.add_argument("--strategy", default="demo_ma")

    args = parser.parse_args()

    if args.cmd == "backtest":
        if args.strategy != "demo_ma":
            logger.error("Only demo_ma is implemented in v1 scaffold.")
            sys.exit(1)
        strat = DemoMAStrategy()
        run_backtest(
            symbol=args.symbol,
            start=_parse_date(args.start),
            end=_parse_date(args.end),
            strategy=strat,
        )
    elif args.cmd == "paper":
        logger.info("Paper trading not implemented yet. Coming soon.")
    elif args.cmd == "live":
        logger.info("Live trading not implemented yet. Coming soon.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Parameter sweep utility for spot backtests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

from bot_v2.backtest import (
    BollingerMeanReversionStrategy,
    MovingAverageCrossStrategy,
    SpotBacktestConfig,
    SpotBacktester,
    load_candles_from_parquet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter sweeps for spot strategies")
    parser.add_argument("parquet", type=Path, help="Candles parquet file")
    parser.add_argument("--strategy", choices=["ma", "bollinger"], default="ma")
    parser.add_argument("--output", type=Path, default=Path("results/sweeps.csv"))
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--short", nargs="*", type=int, default=[5, 10, 15, 20])
    parser.add_argument("--long", nargs="*", type=int, default=[20, 30, 40, 60])
    parser.add_argument("--boll-window", nargs="*", type=int, default=[10, 20, 30])
    parser.add_argument("--boll-std", nargs="*", type=float, default=[1.5, 2.0, 2.5])
    return parser.parse_args()


def sweep_ma(bars, short_list: Iterable[int], long_list: Iterable[int], config: SpotBacktestConfig):
    for short in short_list:
        for long in long_list:
            if short >= long:
                continue
            strategy = MovingAverageCrossStrategy(short_window=short, long_window=long)
            backtester = SpotBacktester(bars, strategy, config)
            result = backtester.run()
            yield {
                "strategy": "ma",
                "short_window": short,
                "long_window": long,
                "total_return": result.metrics.total_return,
                "annualized_return": result.metrics.annualized_return,
                "max_drawdown": result.metrics.max_drawdown,
                "sharpe": result.metrics.sharpe_ratio,
                "trades": len(result.trades),
            }


def sweep_bollinger(bars, window_list: Iterable[int], std_list: Iterable[float], config: SpotBacktestConfig):
    for window in window_list:
        for std in std_list:
            strategy = BollingerMeanReversionStrategy(window=window, num_std=std)
            backtester = SpotBacktester(bars, strategy, config)
            result = backtester.run()
            yield {
                "strategy": "bollinger",
                "window": window,
                "std": std,
                "total_return": result.metrics.total_return,
                "annualized_return": result.metrics.annualized_return,
                "max_drawdown": result.metrics.max_drawdown,
                "sharpe": result.metrics.sharpe_ratio,
                "trades": len(result.trades),
            }


def write_results(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    bars = load_candles_from_parquet(args.parquet)
    if not bars:
        raise SystemExit(f"No data loaded from {args.parquet}")

    config = SpotBacktestConfig(initial_cash=round(args.initial_cash, 2))

    if args.strategy == "ma":
        rows = list(sweep_ma(bars, args.short, args.long, config))
    else:
        rows = list(sweep_bollinger(bars, args.boll_window, args.boll_std, config))

    write_results(args.output, rows)
    print(f"Wrote {len(rows)} results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

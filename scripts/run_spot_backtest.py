#!/usr/bin/env python3
"""Run a spot backtest using stored parquet candles."""

from __future__ import annotations

import argparse
from pathlib import Path

from bot_v2.backtest import (
    BollingerMeanReversionStrategy,
    MovingAverageCrossStrategy,
    SpotBacktestConfig,
    SpotBacktester,
    VolatilityFilteredStrategy,
    VolumeConfirmationStrategy,
    MomentumOscillatorStrategy,
    TrendStrengthStrategy,
    load_candles_from_parquet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spot backtest with stored data")
    parser.add_argument("parquet", type=Path, help="Path to candles parquet file")
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--short-window", type=int, default=12)
    parser.add_argument("--long-window", type=int, default=26)
    parser.add_argument("--strategy", choices=["ma", "bollinger"], default="ma")
    parser.add_argument("--boll-window", type=int, default=20)
    parser.add_argument("--boll-std", type=float, default=2.0)
    parser.add_argument("--vol-window", type=int, default=None, help="Volatility filter ATR window")
    parser.add_argument("--vol-min", type=float, default=None, help="Minimum volatility (ATR/price) to allow entries")
    parser.add_argument("--vol-max", type=float, default=None, help="Maximum volatility (ATR/price) to allow entries")
    parser.add_argument("--volma-window", type=int, default=None, help="Volume confirmation window")
    parser.add_argument("--volma-mult", type=float, default=None, help="Volume multiplier threshold")
    parser.add_argument("--rsi-window", type=int, default=None, help="RSI window for momentum filter")
    parser.add_argument("--rsi-overbought", type=float, default=70.0)
    parser.add_argument("--rsi-oversold", type=float, default=30.0)
    parser.add_argument("--trend-window", type=int, default=None, help="Trend strength MA window")
    parser.add_argument("--trend-min-slope", type=float, default=0.0, help="Minimum slope for trend confirmation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bars = load_candles_from_parquet(args.parquet)
    if not bars:
        raise SystemExit(f"No candles loaded from {args.parquet}")

    if args.strategy == "ma":
        base = MovingAverageCrossStrategy(short_window=args.short_window, long_window=args.long_window)
    else:
        base = BollingerMeanReversionStrategy(window=args.boll_window, num_std=args.boll_std)

    strategy = base
    if args.vol_window and args.vol_min is not None and args.vol_max is not None:
        strategy = VolatilityFilteredStrategy(
            base_strategy=strategy,
            window=args.vol_window,
            min_vol=args.vol_min,
            max_vol=args.vol_max,
        )
    if args.volma_window and args.volma_mult is not None:
        strategy = VolumeConfirmationStrategy(
            base_strategy=strategy,
            window=args.volma_window,
            multiplier=args.volma_mult,
        )

    if args.rsi_window:
        strategy = MomentumOscillatorStrategy(
            base_strategy=strategy,
            window=args.rsi_window,
            overbought=args.rsi_overbought,
            oversold=args.rsi_oversold,
        )

    if args.trend_window:
        strategy = TrendStrengthStrategy(
            base_strategy=strategy,
            window=args.trend_window,
            min_slope=args.trend_min_slope,
        )
    config = SpotBacktestConfig(initial_cash=round(args.initial_cash, 2))
    backtester = SpotBacktester(bars, strategy, config)
    result = backtester.run()

    print(f"Total return: {result.metrics.total_return:.2%}")
    print(f"Annualized return: {result.metrics.annualized_return:.2%}")
    print(f"Max drawdown: {result.metrics.max_drawdown:.2%}")
    print(f"Sharpe ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"Trades: {len(result.trades)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run backtests for all symbols defined in a spot profile YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

from bot_v2.backtest import SpotBacktester, load_candles_from_parquet
from bot_v2.backtest.profile import build_strategy_spec, load_profile


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest strategies defined in a spot profile")
    parser.add_argument("profile", type=Path, help="Path to spot profile YAML")
    parser.add_argument("--data-root", type=Path, default=Path("data/spot_raw"), help="Directory containing symbol parquet folders")
    parser.add_argument("--granularity", default="1h", help="Parquet granularity suffix (default: 1h)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = load_profile(args.profile)
    symbols = profile.get("symbols", [])
    if not symbols:
        raise SystemExit("No symbols defined in profile")

    for symbol in symbols:
        spec = build_strategy_spec(args.profile, symbol)
        parquet_path = args.data_root / symbol.replace("-", "_") / f"candles_{args.granularity}.parquet"
        bars = load_candles_from_parquet(parquet_path)
        backtester = SpotBacktester(bars, spec.strategy, spec.config)
        result = backtester.run()
        print(
            f"{symbol} -> total_return={result.metrics.total_return:.2%} "
            f"sharpe={result.metrics.sharpe_ratio:.2f} drawdown={result.metrics.max_drawdown:.2%} "
            f"trades={len(result.trades)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
from datetime import datetime

import numpy as np

from ..dataflow.sources.yfinance_source import YFinanceSource
from ..logging import get_logger
from ..strategy.base import Strategy

logger = get_logger("backtest")


def run_backtest(symbol: str, start: datetime, end: datetime, strategy: Strategy) -> None:
    logger.info(f"Backtesting {strategy.name} on {symbol} from {start.date()} to {end.date()}")

    src = YFinanceSource()
    df = src.get_daily_bars(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    sig = strategy.generate_signals(df)
    data = df.join(sig, how="left").dropna()

    # Simple daily-close execution model:
    # If signal flips from <=0 to >0 -> buy at next open
    # If signal flips from >=0 to <0 -> sell at next open
    data["position"] = 0
    data["position"] = np.where(data["signal"] > 0, 1, np.where(data["signal"] < 0, -1, 0))
    data["position_shift"] = data["position"].shift(1).fillna(0)

    # Returns based on close-to-close
    data["ret"] = data["Close"].pct_change().fillna(0.0)
    # Strategy returns: yesterday's position * today's return
    data["strat_ret"] = data["position_shift"] * data["ret"]

    equity0 = 100000.0
    data["equity"] = equity0 * (1 + data["strat_ret"]).cumprod()

    # Output
    outdir = os.path.join("data", "backtests")
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outfile = os.path.join(outdir, f"{symbol}_{strategy.name}_{stamp}.csv")
    data[["Close", "signal", "position_shift", "ret", "strat_ret", "equity"]].to_csv(
        outfile, index=True
    )

    total_ret = data["equity"].iloc[-1] / equity0 - 1
    cagr = (1 + total_ret) ** (252 / max(1, len(data))) - 1  # rough
    max_dd = ((data["equity"].cummax() - data["equity"]) / data["equity"].cummax()).max()
    # QoL: rough trade count based on position changes
    trade_count = int((data["position"] != data["position_shift"]).sum())
    logger.info("Bars: %d | Trades: %d", len(data), trade_count)
    logger.info(f"Results saved to {outfile} (final_equity={data['equity'].iloc[-1]:.2f})")
    logger.info(f"Total return: {total_ret:.2%} | CAGR~: {cagr:.2%} | Max DD: {max_dd:.2%}")

    logger.info("Backtest complete.")

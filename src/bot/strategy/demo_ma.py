from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


def _wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    # True range components
    hl = (df["High"] - df["Low"]).abs()
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    # Simple rolling mean is fine for now (Wilder's RMA could be added later)
    return tr.rolling(window=period, min_periods=period).mean()


class DemoMAStrategy(Strategy):
    name = "demo_ma"

    def __init__(self, fast: int = 10, slow: int = 20, atr_period: int = 14) -> None:
        self.fast = int(fast)
        self.slow = int(slow)
        self.atr_period = int(atr_period)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Moving averages off Close; strict windows (no look-ahead)
        sma_fast = df["Close"].rolling(self.fast, min_periods=self.fast).mean()
        sma_slow = df["Close"].rolling(self.slow, min_periods=self.slow).mean()

        sig = np.where(sma_fast > sma_slow, 1.0, np.where(sma_fast < sma_slow, -1.0, 0.0))

        # Long-only version (flip to 1/0). If you want long/short, remove the clip.
        sig = np.clip(sig, 0, 1)

        atr = _wilder_atr(df, self.atr_period)

        out = pd.DataFrame(
            {
                "signal": sig,
                "sma_fast": sma_fast,
                "sma_slow": sma_slow,
                "atr": atr,
            },
            index=df.index,
        )

        # Zero out signals until both MAs and ATR are ready
        ready_mask = (~out["sma_fast"].isna()) & (~out["sma_slow"].isna()) & (~out["atr"].isna())
        out.loc[~ready_mask, "signal"] = 0.0

        return out

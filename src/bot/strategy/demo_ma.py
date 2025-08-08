from __future__ import annotations

import pandas as pd

from .base import Strategy


class DemoMAStrategy(Strategy):
    name = "demo_ma"

    def __init__(self, fast: int = 10, slow: int = 20) -> None:
        if fast >= slow:
            raise ValueError("fast MA must be < slow MA")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        df["ma_fast"] = df["Close"].rolling(self.fast).mean()
        df["ma_slow"] = df["Close"].rolling(self.slow).mean()
        df["signal"] = 0
        df.loc[df["ma_fast"] > df["ma_slow"], "signal"] = 1
        df.loc[df["ma_fast"] < df["ma_slow"], "signal"] = -1
        return df[["signal", "ma_fast", "ma_slow"]]

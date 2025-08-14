from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from bot.indicators.atr import atr
from bot.indicators.donchian import donchian_channels

from .base import Strategy


@dataclass
class TrendBreakoutParams:
    donchian_lookback: int = 55
    atr_period: int = 20
    atr_k: float = 2.0  # used for sizing/exits later


class TrendBreakoutStrategy(Strategy):
    name = "trend_breakout"
    supports_short = False

    def __init__(self, params: TrendBreakoutParams | None = None) -> None:
        self.params = params or TrendBreakoutParams()

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        upper, lower = donchian_channels(df, self.params.donchian_lookback)
        a = atr(df, self.params.atr_period, method="wilder")

        # Long when close breaks above prior-day upper channel
        df["breakout_long"] = (df["Close"] > upper.shift(1)).astype(int)
        df["signal"] = 0
        df.loc[df["breakout_long"] == 1, "signal"] = 1

        df["donchian_upper"] = upper
        df["donchian_lower"] = lower
        df["atr"] = a
        return df[["signal", "donchian_upper", "donchian_lower", "atr"]]

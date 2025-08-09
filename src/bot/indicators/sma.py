from __future__ import annotations

import pandas as pd


def sma(series: pd.Series | list | tuple, window: int) -> pd.Series:
    # Generic SMA (min_periods=1). For regimes, we’ll use strict SMA inside the engine.
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    return series.rolling(window=window, min_periods=1).mean()

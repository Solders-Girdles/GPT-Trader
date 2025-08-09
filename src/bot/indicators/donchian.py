from __future__ import annotations

import pandas as pd


def donchian_channels(df: pd.DataFrame, lookback: int = 55) -> tuple[pd.Series, pd.Series]:
    upper = df["High"].rolling(lookback, min_periods=lookback).max()
    lower = df["Low"].rolling(lookback, min_periods=lookback).min()
    return upper, lower

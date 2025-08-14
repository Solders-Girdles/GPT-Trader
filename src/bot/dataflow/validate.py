from __future__ import annotations

import numpy as np
import pandas as pd
from bot.utils.validation import DataFrameValidator


def adjust_to_adjclose(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    If an 'Adj Close' column exists, adjust OHLC to the adjusted close
    by applying the daily factor: f = AdjClose / Close.
    Returns (df_adjusted, did_adjust).
    """
    if "Adj Close" not in df.columns or "Close" not in df.columns:
        return df, False

    close = df["Close"].astype(float)
    adj = df["Adj Close"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = adj / close

    # If factor is effectively constant (≈1), skip to avoid needless FP noise
    if factor.dropna().std() < 1e-6:
        return df, False

    out = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            out[col] = out[col].astype(float) * factor

    return out, True


def validate_daily_bars(df: pd.DataFrame, symbol: str) -> None:
    """
    Lightweight sanity checks for daily bars.
    Raises ValueError on hard failures; prints warnings for soft issues.
    """
    # Use consolidated validation
    DataFrameValidator.validate_daily_bars(df, symbol)

    # Additional soft checks (warn only) — Long gaps: > 7 calendar days between rows
    gaps = df.index.to_series().diff().dt.days.dropna()
    long_gaps = gaps[gaps > 7]
    if len(long_gaps) > 0:
        first_gap = int(long_gaps.iloc[0])
        print(f"WARNING {symbol}: detected long date gap (first gap {first_gap} days)")

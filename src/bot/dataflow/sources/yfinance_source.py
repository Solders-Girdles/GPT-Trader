from __future__ import annotations

import pandas as pd
import yfinance as yf

from .. import base as base


class YFinanceSource(base.HistoricalDataSource):
    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        # Normalize column names
        df = df.rename(
            columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Adj Close": "AdjClose",
                "Volume": "Volume",
            }
        )
        # Ensure expected cols exist
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns from yfinance for {symbol}: {missing}")
        return df

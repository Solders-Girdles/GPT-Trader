from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf


# --- Local Parquet cache for yfinance ---
def _cache_dir() -> Path:
    d = Path("data/cache/yf")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(symbol: str) -> Path:
    return _cache_dir() / f"{symbol.upper()}.parquet"


def _read_parquet_safe(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _write_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=True)
    except Exception:
        # Fall back to CSV if parquet engine not available
        path_csv = path.with_suffix(".csv")
        df.to_csv(path_csv, index=True)


def _normalize_symbol(sym: str) -> str:
    # Yahoo uses hyphens for share classes (BRK.B -> BRK-B)
    return sym.replace(".", "-").upper().strip()


@dataclass
class YFinanceSource:
    auto_adjust: bool = True
    retries: int = 3
    retry_sleep: float = 1.0  # seconds
    timeout: int = 30

    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        sym = _normalize_symbol(symbol)
        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                t = yf.Ticker(sym)
                df = t.history(
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=self.auto_adjust,
                    actions=False,
                    timeout=self.timeout,
                )
                # yfinance sometimes returns empty or missing cols; enforce schema
                if df is None or df.empty:
                    raise RuntimeError(f"Empty data for {sym}")
                req = ["Open", "High", "Low", "Close", "Volume"]
                for c in req:
                    if c not in df.columns:
                        raise RuntimeError(f"Missing column {c} for {sym}")
                df = df[req]
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                return df
            except Exception as e:  # transient network hiccups etc.
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.retry_sleep * attempt)
                else:
                    raise

        # Shouldn’t get here; keep mypy happy
        assert last_err is not None
        raise last_err

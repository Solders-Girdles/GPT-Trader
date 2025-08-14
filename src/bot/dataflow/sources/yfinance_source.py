from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from bot.logging import get_logger

logger = get_logger("data")


# --- Local Parquet/CSV cache for yfinance ---
def _cache_dir() -> Path:
    d = Path(os.getenv("YF_CACHE_DIR", "data/cache/yf"))
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

    def get_daily_bars(
        self, symbol: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Cached wrapper around `_get_daily_bars_uncached`.

        Opt-in via env:
          - YF_CACHE=1 (enable)
          - YF_CACHE_DIR=path (default data/cache/yf)
          - YF_CACHE_TTL_DAYS=N (default 30)
        """
        use_cache = str(os.getenv("YF_CACHE", "1")).lower() not in ("0", "false", "off", "no")
        if not use_cache:
            fs = start or ""
            fe = end or ""
            if fe:
                try:
                    fe_dt = pd.to_datetime(fe) + pd.Timedelta(days=1)
                    fe = fe_dt.strftime("%Y-%m-%d")
                except Exception as e:
                    logger.debug(f"Failed to parse end date '{fe}': {e}")
            return self._get_daily_bars_uncached(symbol, start=fs, end=fe)

        cache_dir = os.getenv("YF_CACHE_DIR", str(_cache_dir()))
        ttl_days = int(os.getenv("YF_CACHE_TTL_DAYS", "30"))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        pq_path = Path(cache_dir) / f"{symbol.upper()}.parquet"
        csv_path = pq_path.with_suffix(".csv")

        def _norm(df: pd.DataFrame | None) -> pd.DataFrame:
            """Normalize yfinance frame to DateTimeIndex and sorted ascending."""
            if df is None or len(df) == 0:
                return pd.DataFrame()
            if not isinstance(df.index, pd.DatetimeIndex):
                if "Date" in df.columns:
                    df = df.set_index("Date")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            df.index.name = "Date"
            return df

        req_start = pd.to_datetime(start) if start else None
        req_end = pd.to_datetime(end) if end else None

        # Load cache if present (prefer Parquet, fallback to CSV)
        cache = pd.DataFrame()
        cache = _read_parquet_safe(pq_path)
        cache = _norm(cache)
        if (cache is None or cache.empty) and csv_path.exists():
            try:
                cache = _norm(pd.read_csv(csv_path, index_col=0))
            except Exception:
                cache = pd.DataFrame()

        # Determine staleness (prefer Parquet, fallback to CSV)
        stale = False
        file_for_mtime = pq_path if pq_path.exists() else (csv_path if csv_path.exists() else None)
        if file_for_mtime and ttl_days > 0:
            try:
                mtime = datetime.fromtimestamp(file_for_mtime.stat().st_mtime)
                stale = (datetime.now() - mtime) > timedelta(days=ttl_days)
            except Exception:
                stale = True

        if not cache.empty and not stale:
            have_first = cache.index.min()
            have_last = cache.index.max()
            need_start = req_start or have_first
            need_end = req_end or have_last
            if have_first <= need_start and have_last >= need_end:
                return cache.loc[need_start:need_end].copy()

        # Determine fetch window
        fetch_start = req_start
        fetch_end = req_end
        if fetch_start is None and not cache.empty:
            fetch_start = cache.index.min()
        if fetch_end is None and not cache.empty:
            fetch_end = cache.index.max()

        fs = fetch_start.strftime("%Y-%m-%d") if fetch_start is not None else ""
        fe = fetch_end.strftime("%Y-%m-%d") if fetch_end is not None else ""
        if fe:
            try:
                fe_dt = pd.to_datetime(fe) + pd.Timedelta(days=1)
                fe = fe_dt.strftime("%Y-%m-%d")
            except Exception as e:
                logger.debug(f"Failed to parse end date '{fe}': {e}")

        fresh = self._get_daily_bars_uncached(symbol, start=fs, end=fe)
        fresh = _norm(fresh)

        merged = pd.concat([cache, fresh]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]

        # Best-effort cache write
        try:
            _write_parquet_safe(merged, pq_path)
        except Exception as e:  # best-effort cache write
            logger.debug(f"Cache write failed for {pq_path}: {e}")

        if req_start is None and req_end is None:
            return merged.copy()
        lo = req_start or merged.index.min()
        hi = req_end or merged.index.max()
        return merged.loc[lo:hi].copy()

    def _get_daily_bars_uncached(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        sym = _normalize_symbol(symbol)
        last_err: Exception | None = None
        req_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = None
        # Widen fetch window by 2 days on both sides to avoid boundary issues
        try:
            start_dt = pd.to_datetime(start) if start else None
            end_dt = pd.to_datetime(end) if end else None
            if start_dt is not None:
                start_dt = start_dt - pd.Timedelta(days=2)
            if end_dt is not None:
                end_dt = end_dt + pd.Timedelta(days=2)
            start_str = start_dt.strftime("%Y-%m-%d") if start_dt is not None else ""
            end_str = end_dt.strftime("%Y-%m-%d") if end_dt is not None else ""
        except Exception:
            start_str = start or ""
            end_str = end or ""

        for attempt in range(1, self.retries + 1):
            try:
                t = yf.Ticker(sym)
                df = t.history(
                    start=start_str,
                    end=end_str,
                    interval="1d",
                    auto_adjust=self.auto_adjust,
                    actions=False,
                    timeout=self.timeout,
                )
                # yfinance sometimes returns empty or missing cols; enforce schema
                if df is None or df.empty:
                    raise RuntimeError(f"Empty data for {sym}")
                for c in req_cols:
                    if c not in df.columns:
                        raise RuntimeError(f"Missing column {c} for {sym}")
                df = df[req_cols]
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                df = df[~df.index.duplicated(keep="last")]
                return df
            except Exception as e:
                err_msg = str(e).lower()
                # Handle "possibly delisted" and "no price data found" errors
                if (
                    ("possibly delisted" in err_msg)
                    or ("no price data found" in err_msg)
                    or (df is None or df.empty)
                ):
                    logger.warning(f"YFinance data fetch warning for {sym}: {e}")
                    empty_df = pd.DataFrame({c: pd.Series(dtype="float64") for c in req_cols})
                    empty_df.index.name = "Date"
                    return empty_df
                # Handle transient or rate limit errors with retry
                if (
                    "too many requests" in err_msg
                    or "rate limit" in err_msg
                    or "timed out" in err_msg
                    or "temporarily unavailable" in err_msg
                ):
                    if attempt < self.retries:
                        sleep_time = (
                            self.retry_sleep * (2 ** (attempt - 1)) * (1 + random.random() * 0.5)
                        )
                        time.sleep(sleep_time)
                        continue
                    else:
                        raise
                last_err = e
                if attempt < self.retries:
                    # shorter backoff for transient errors
                    sleep_time = self.retry_sleep * 0.5 * attempt * (1 + random.random() * 0.25)
                    time.sleep(sleep_time)
                else:
                    raise
        assert last_err is not None
        raise last_err

    @staticmethod
    def warm_cache(symbols: list[str], start: str, end: str, quiet: bool = False) -> None:
        yfs = YFinanceSource()
        for sym in symbols:
            try:
                df = yfs.get_daily_bars(sym, start=start, end=end)
                if not quiet:
                    if df.empty:
                        logger.info(f"Warm cache: {sym} returned no data")
                    else:
                        first_date = df.index.min().strftime("%Y-%m-%d")
                        last_date = df.index.max().strftime("%Y-%m-%d")
                        logger.info(
                            f"Warm cache: {sym} rows={len(df)} from {first_date} to {last_date}"
                        )
            except Exception as e:
                logger.warning(f"Warm cache: failed to fetch {sym}: {e}")

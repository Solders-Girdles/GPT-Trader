from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bot.config import get_config
from bot.dataflow.sources.yfinance_source import YFinanceSource
from bot.dataflow.validate import adjust_to_adjclose, validate_daily_bars
from bot.exec.ledger import Ledger
from bot.logging import get_logger
from bot.metrics.report import perf_metrics
from bot.portfolio.allocator import PortfolioRules, allocate_signals
from bot.strategy.base import Strategy

T = TypeVar("T")
logger = get_logger("backtest")


@dataclass
class BacktestData:
    """Reusable, pre-aligned data bundle to speed multi-run optimization."""

    symbols: list[str]
    data_map: dict[str, pd.DataFrame]
    regime_ok: pd.Series | None
    dates_idx: pd.DatetimeIndex


def _iter_progress(seq: Iterable[T], desc: str = "backtest") -> Iterator[T]:
    """Yield items from `seq` with a tqdm progress bar if available."""
    try:
        from tqdm import tqdm  # type: ignore

        it = tqdm(seq, desc=desc, leave=False)
    except Exception:
        it = seq
    yield from it


def _warn(msg: str) -> None:
    logger.warning(msg)


def validate_ohlc(df: pd.DataFrame, symbol: str, strict_mode: bool = True) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{symbol}: empty DataFrame after load/adjust")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{symbol}: index must be DatetimeIndex, got {type(df.index)}")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: DatetimeIndex is not sorted ascending")

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{symbol}: missing required columns: {missing}")

    nan_ct = int(df[required].isna().sum().sum())
    if nan_ct > 0:
        raise ValueError(f"{symbol}: NaNs in OHLC ({nan_ct})")

    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    low = df["Low"].astype(float)
    c = df["Close"].astype(float)
    max_oc = np.maximum(o.values, c.values)
    min_oc = np.minimum(o.values, c.values)
    if bool((h.values < max_oc).any()) or bool((low.values > min_oc).any()):
        raise ValueError(f"{symbol}: invalid OHLC bounds")

    gaps = df.index.to_series().diff().dt.days.dropna()
    long_gaps = gaps[gaps > 7]
    if len(long_gaps) > 0:
        _warn(f"{symbol}: detected long date gap (first gap {int(long_gaps.iloc[0])} days)")

    non_bday = df.index[~df.index.to_series().dt.dayofweek.isin(range(5))]
    if len(non_bday) > 0:
        _warn(f"{symbol}: {len(non_bday)} rows on non-business days")

    for col in required:
        if (df[col] <= 0).any():
            _warn(f"{symbol}: non-positive values found in {col}")
    if "Volume" in df.columns and (df["Volume"] < 0).any():
        _warn(f"{symbol}: negative Volume values found")

    return df


# --- Minimal BacktestEngine adapter for training/validation code ---


@dataclass
class BacktestConfig:
    """Lightweight configuration for the simplified backtest engine.

    This adapter exists to satisfy components that expect an OO backtest API.
    """

    start_date: datetime
    end_date: datetime
    initial_capital: float = None  # Will use config default if not specified
    commission_rate: float = None  # Will use config default if not specified

    def __post_init__(self) -> None:
        """Load defaults from unified configuration if not specified."""
        config = get_config()
        if self.initial_capital is None:
            self.initial_capital = float(config.financial.capital.backtesting_capital)
        if self.commission_rate is None:
            self.commission_rate = config.financial.costs.commission_rate_decimal


class BacktestEngine:
    """Minimal engine that runs a simple next-bar backtest over a single DataFrame.

    The goal is compatibility with higher-level training/validation flows that
    expect a class-based API returning a metrics dict.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run_backtest(self, strategy: Strategy, data: pd.DataFrame) -> dict[str, object]:
        if data.empty:
            return {
                "metrics": {
                    "sharpe_ratio": 0.0,
                    "calmar_ratio": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "sortino_ratio": 0.0,
                }
            }

        # Subset to config date range if present in the data index
        df = data.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.loc[(df.index >= self.config.start_date) & (df.index <= self.config.end_date)]
            if df.empty:
                return {
                    "metrics": {
                        "sharpe_ratio": 0.0,
                        "calmar_ratio": 0.0,
                        "total_return": 0.0,
                        "max_drawdown": 0.0,
                        "sortino_ratio": 0.0,
                    }
                }

        # Generate signals and build simple position series (allowing -1/0/1)
        sig = strategy.generate_signals(df)
        merged = df.join(sig, how="left")
        position = merged.get("signal", pd.Series(0, index=merged.index)).fillna(0.0)
        position = position.clip(-1, 1)
        position_shift = position.shift(1).fillna(0.0)

        # Close-to-close returns
        ret = merged["Close"].pct_change().fillna(0.0)

        # Apply simple transaction costs on position changes (per unit change)
        if self.config.commission_rate and self.config.commission_rate > 0:
            pos_change = position.abs() - position_shift.abs()
            trading_cost = (pos_change.abs() * self.config.commission_rate).fillna(0.0)
        else:
            trading_cost = pd.Series(0.0, index=ret.index)

        strat_ret = position_shift * ret - trading_cost

        equity0 = float(self.config.initial_capital)
        equity = equity0 * (1.0 + strat_ret).cumprod()

        # Metrics
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        ann = 252
        vol = float(strat_ret.std() * np.sqrt(ann))
        sharpe = 0.0 if vol == 0.0 else float(strat_ret.mean() * ann / vol)
        roll_max = equity.cummax()
        dd = (roll_max - equity) / roll_max
        max_dd = float(dd.max()) if len(dd) else 0.0
        cagr = (1.0 + total_return) ** (ann / max(1, len(equity))) - 1.0
        calmar = 0.0 if max_dd == 0.0 else float(cagr / max_dd)
        downside = strat_ret[strat_ret < 0].std() * np.sqrt(ann)
        sortino = (
            0.0
            if (downside is None or downside == 0)
            else float((strat_ret.mean() * ann) / downside)
        )

        metrics = {
            "sharpe_ratio": float(sharpe),
            "calmar_ratio": float(calmar),
            "total_return": float(total_return),
            "max_drawdown": float(max_dd),
            "sortino_ratio": float(sortino),
        }

        return {"metrics": metrics, "equity": equity}


def _read_universe_csv(path: str) -> list[str]:
    df = pd.read_csv(path)
    col: str | None = None
    for c in df.columns:
        if c.strip().lower() in {"symbol", "ticker"}:
            col = c
            break
    if col is None:
        raise ValueError("Universe CSV must have a column named 'symbol' or 'ticker'")
    syms = [str(s).strip().upper() for s in df[col].dropna().unique().tolist()]
    return syms


def _safe_float(x: pd.Series | float | int) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    if hasattr(x, "iloc"):
        return float(x.iloc[0])
    return float(x)


def _strict_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def prepare_backtest_data(
    symbol: str | None,
    symbol_list_csv: str | None,
    start: datetime,
    end: datetime,
    regime_on: bool = False,
    regime_symbol: str = "SPY",
    regime_window: int = 200,
    strict_mode: bool = True,
    warmup_bars_symbols: int = 260,
    warmup_bars_regime_extra: int = 60,
    *,
    symbols: list[str] | None = None,
) -> BacktestData:
    """
    Load, adjust, validate, and align universe + optional regime series once.
    Returns a BacktestData bundle that can be reused across many runs/parameters.
    """
    src = YFinanceSource()

    if symbols is not None and len(symbols) > 0:
        symbols = [str(s).strip().upper() for s in symbols]
    elif symbol_list_csv:
        symbols = _read_universe_csv(symbol_list_csv)
    elif symbol:
        symbols = [symbol]
    else:
        raise ValueError("Provide symbols list, --symbol, or --symbol-list")

    # Warm-ups to satisfy indicators/regime
    pad_days_symbols = int(warmup_bars_symbols)
    pad_days_regime = (regime_window + int(warmup_bars_regime_extra)) if regime_on else 0

    fetch_start_symbols = start - timedelta(days=pad_days_symbols)
    fetch_start_regime = start - timedelta(days=pad_days_regime) if regime_on else None

    eval_start_ts = pd.Timestamp(start.date())
    eval_end_ts = pd.Timestamp(end.date())

    # Load symbol data
    data_map: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for sym in symbols:
        try:
            df = src.get_daily_bars(
                sym,
                start=fetch_start_symbols.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
            df_adj, _ = adjust_to_adjclose(df)
            df_adj = validate_ohlc(df_adj, sym, strict_mode=strict_mode)
            validate_daily_bars(df_adj, sym)
            df = df_adj
            df.index = pd.to_datetime(df.index).tz_localize(None)
            data_map[sym] = df
        except Exception as e:
            logger.warning(f"Skipping {sym}: {e}")
            skipped.append(sym)

    if not data_map:
        raise RuntimeError("No data loaded in prepare_backtest_data")

    # Optional regime
    regime_ok: pd.Series | None = None
    if regime_on:
        try:
            mkt = src.get_daily_bars(
                regime_symbol,
                start=(
                    fetch_start_regime.strftime("%Y-%m-%d")
                    if fetch_start_regime is not None
                    else start.strftime("%Y-%m-%d")
                ),
                end=end.strftime("%Y-%m-%d"),
            )
            mkt_adj, _ = adjust_to_adjclose(mkt)
            mkt_adj = validate_ohlc(mkt_adj, regime_symbol, strict_mode=strict_mode)
            validate_daily_bars(mkt_adj, regime_symbol)
            mkt = mkt_adj
            mkt.index = pd.to_datetime(mkt.index).tz_localize(None)
            ma = _strict_sma(mkt["Close"], regime_window)
            regime_ok = (mkt["Close"] > ma).astype(bool)
        except Exception as e:
            logger.warning(f"Regime fetch failed in prepare_backtest_data ({regime_symbol}): {e}")
            regime_ok = None

    # Master dates (union across symbols), then crop to [start, end]
    all_dates_raw = sorted(set().union(*[df.index for df in data_map.values()]))
    all_dates = [
        pd.Timestamp(d).tz_localize(None) if getattr(d, "tz", None) is not None else pd.Timestamp(d)
        for d in all_dates_raw
    ]
    dates = [d for d in all_dates if (d >= eval_start_ts and d <= eval_end_ts)]
    if len(dates) < 2:
        raise RuntimeError("Not enough dates to build aligned index in prepare_backtest_data")
    dates_idx = pd.DatetimeIndex(dates, name="Date")

    return BacktestData(
        symbols=list(data_map.keys()),
        data_map=data_map,
        regime_ok=(regime_ok.reindex(dates_idx).fillna(True) if regime_ok is not None else None),
        dates_idx=dates_idx,
    )


def run_backtest(
    symbol: str | None,
    symbol_list_csv: str | None,
    start: datetime,
    end: datetime,
    strategy: Strategy,
    rules: PortfolioRules,
    regime_on: bool = False,
    regime_symbol: str = "SPY",
    regime_window: int = 200,
    debug: bool = False,
    exit_mode: Literal["signal", "stop"] = "signal",
    cadence: Literal["daily", "weekly"] = "daily",
    cooldown: int = 0,
    entry_confirm: int = 1,
    warmup_bars_symbols: int = 260,
    warmup_bars_regime_extra: int = 60,
    min_rebalance_pct: float = 0.0,
    strict_mode: bool = True,
    show_progress: bool = False,
    make_plot: bool = True,
    write_portfolio_csv: bool = True,
    write_trades_csv: bool = True,
    write_summary_csv: bool = True,
    progress_desc: str = "days",
    quiet_mode: bool = False,
    *,
    prepared: BacktestData | None = None,
    return_summary: bool = False,
    return_equity: bool = False,
    return_trades: bool = False,
) -> dict[str, object] | None:
    """
    v1.5 timing model + ATR trailing exits:
    - Signals use data through D-1.
    - Trades execute at D open (price + costs at Open[D]).
    - PnL = overnight (Close[D-1]→Open[D]) on old holdings
      + intraday (Open[D]→Close[D]) on new holdings.
    - Regime filter (SPY > 200DMA) evaluated at D-1 (strict SMA).
    - ATR trailing stop: computed from D-1 Close/ATR, ratchets up only.
      If breached at D-1, exit at D open.
    Returns (optional): when any of `return_summary`, `return_equity`, or `return_trades` is True, a dict is returned with keys among {"summary", "equity", "trades"}. Callers that ignore the return value are unaffected.
    """
    # Check for quiet mode environment variable
    if os.getenv("BACKTEST_QUIET", "0") == "1":
        quiet_mode = True

    if not quiet_mode:
        logger.info(f"Backtesting {strategy.name} from {start.date()} to {end.date()}")
        logger.info(f"Data strict mode: {'strict' if strict_mode else 'repair'}")
    outdir = os.path.join("data", "backtests")
    os.makedirs(outdir, exist_ok=True)

    src = YFinanceSource()

    if symbol_list_csv:
        symbols = _read_universe_csv(symbol_list_csv)
    elif symbol:
        symbols = [symbol]
    else:
        raise ValueError("Provide --symbol or --symbol-list")

    # If we were given a prepared bundle, reuse it and skip all I/O/alignment work
    if prepared is not None:
        data_map = prepared.data_map
        symbols = [s for s in prepared.symbols if s in data_map]
        if not data_map:
            logger.error("Prepared data bundle is empty; aborting.")
            return
        dates_idx = prepared.dates_idx
        dates = list(dates_idx.to_pydatetime())
        regime_arr = (
            prepared.regime_ok.to_numpy(dtype=bool, copy=False)  # type: ignore[union-attr]
            if (prepared.regime_ok is not None)
            else np.ones(len(dates_idx), dtype=bool)
        )
    else:
        # --- Warm-up windows so indicators/regime have history ---
        pad_days_symbols = int(warmup_bars_symbols)  # ~1 trading year default
        pad_days_regime = (regime_window + int(warmup_bars_regime_extra)) if regime_on else 0

        fetch_start_symbols = start - timedelta(days=pad_days_symbols)
        fetch_start_regime = start - timedelta(days=pad_days_regime) if regime_on else None

        # Cut window we actually evaluate PnL on
        eval_start_ts = pd.Timestamp(start.date())
        eval_end_ts = pd.Timestamp(end.date())

        logger.debug(
            f"Warm-up: symbols from {fetch_start_symbols.date()} (pad {pad_days_symbols}d); "
            f"regime from {(fetch_start_regime.date() if fetch_start_regime else start.date())} "
            f"(pad {pad_days_regime}d)"
        )

        # 1) Load data
        data_map: dict[str, pd.DataFrame] = {}
        skipped_syms: list[str] = []
        for sym in symbols:
            try:
                df = src.get_daily_bars(
                    sym,
                    start=fetch_start_symbols.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                )
                # --- Data adjustments & validation ---
                df_adj, did_adj = adjust_to_adjclose(df)
                # Validate after adjustment to catch gaps/NaNs/bounds
                df_adj = validate_ohlc(df_adj, sym, strict_mode=strict_mode)
                if did_adj and not quiet_mode:
                    logger.info(f"{sym}: applied OHLC adjustment using Adj Close factors")
                validate_daily_bars(df_adj, sym)
                df = df_adj
                # Ensure tz-naive index for safe comparisons
                df.index = pd.to_datetime(df.index).tz_localize(None)
                data_map[sym] = df
            except Exception as e:
                logger.warning(f"Skipping {sym}: {e}")
                skipped_syms.append(sym)

        if skipped_syms and not quiet_mode:
            logger.info(
                f"Skipped {len(skipped_syms)} symbol(s) with bad/empty data: {', '.join(skipped_syms[:8])}{'…' if len(skipped_syms) > 8 else ''}"
            )

        if not data_map:
            logger.error("No data loaded; aborting.")
            return

        # 2) Regime series (optional)
        regime_ok: pd.Series | None = None
        if regime_on:
            try:
                mkt = src.get_daily_bars(
                    regime_symbol,
                    start=(
                        fetch_start_regime.strftime("%Y-%m-%d")
                        if fetch_start_regime is not None
                        else start.strftime("%Y-%m-%d")
                    ),
                    end=end.strftime("%Y-%m-%d"),
                )
                # Adjust/validate regime series the same as symbols
                mkt_adj, _ = adjust_to_adjclose(mkt)
                mkt_adj = validate_ohlc(mkt_adj, regime_symbol, strict_mode=strict_mode)
                validate_daily_bars(mkt_adj, regime_symbol)
                mkt = mkt_adj
                # Ensure tz-naive index for safe comparisons
                mkt.index = pd.to_datetime(mkt.index).tz_localize(None)
                ma = _strict_sma(mkt["Close"], regime_window)
                regime_ok = (mkt["Close"] > ma).astype(bool)
            except Exception as e:
                logger.warning(f"Regime fetch failed ({regime_symbol}): {e}")
                regime_ok = None

        # Build the evaluation date range from loaded data, but only keep [start, end]
        all_dates_raw = sorted(set().union(*[df.index for df in data_map.values()]))
        # Normalize to tz-naive to avoid comparing tz-aware vs tz-naive
        all_dates = [
            (
                pd.Timestamp(d).tz_localize(None)
                if getattr(d, "tz", None) is not None
                else pd.Timestamp(d)
            )
            for d in all_dates_raw
        ]
        dates = [d for d in all_dates if (d >= eval_start_ts and d <= eval_end_ts)]
        if len(dates) < 2:
            logger.error("Not enough dates to run a next-open backtest.")
            return
        dates_idx = pd.DatetimeIndex(dates, name="Date")

        # Regime array aligned to master dates
        if regime_on and regime_ok is not None:
            regime_arr = regime_ok.reindex(dates_idx).fillna(True).to_numpy(dtype=bool, copy=False)
        else:
            regime_arr = np.ones(len(dates_idx), dtype=bool)

    # 3) Signals per symbol
    sigs: dict[str, pd.DataFrame] = {}
    for sym, df in data_map.items():
        s = strategy.generate_signals(df)
        sigs[sym] = df.join(s, how="left")

    # --- Fast-path caches aligned to a common date index ---
    sym_cache: dict[str, dict[str, object]] = {}
    for sym, df in sigs.items():
        # Reindex each symbol's frame to the master dates to enable O(1) positional access
        r = df.reindex(dates_idx)
        # Precompute arrays (numpy for speed)
        open_arr = r["Open"].to_numpy(dtype=float, copy=False)
        close_arr = r["Close"].to_numpy(dtype=float, copy=False)
        signal_arr = (
            r["signal"].to_numpy(dtype=float, copy=False)
            if "signal" in r.columns
            else np.full(len(r), np.nan, dtype=float)
        )
        atr_arr = (
            r["atr"].to_numpy(dtype=float, copy=False)
            if "atr" in r.columns
            else np.full(len(r), np.nan, dtype=float)
        )
        don_arr = (
            r["donchian_upper"].to_numpy(dtype=float, copy=False)
            if "donchian_upper" in r.columns
            else np.full(len(r), np.nan, dtype=float)
        )

        # Entry confirmation helper (True iff last N signals strictly positive at position i)
        if entry_confirm and entry_confirm > 1 and "signal" in r.columns:
            pos_signal = (r["signal"] > 0).astype(int)
            # rolling sum equals window size => all > 0
            entry_ok = (
                pos_signal.rolling(window=entry_confirm, min_periods=entry_confirm).sum()
                == entry_confirm
            ).to_numpy(dtype=bool, copy=False)
        else:
            entry_ok = np.ones(len(r), dtype=bool)

        sym_cache[sym] = {
            "frame": r,  # reindexed DataFrame for occasional slices
            "open": open_arr,
            "close": close_arr,
            "signal": signal_arr,
            "atr": atr_arr,
            "don": don_arr,
            "entry_ok": entry_ok,
        }

    # Build the evaluation date range from loaded data, but only keep [start, end]
    # (removed duplicate block; see earlier in function)

    # 4) State
    config = get_config()
    equity0 = float(config.financial.capital.backtesting_capital)
    n_dates = len(dates)
    equity_arr = np.empty(n_dates, dtype=float)
    equity_arr[:] = np.nan
    equity = equity0
    holdings: dict[str, int] = {}
    # ensure name exists for static checkers; populated each loop
    desired: dict[str, int] = {}

    total_costs = 0.0
    ledger = Ledger()

    # track last exit date (decision date prev_dt) per symbol for cooldown
    last_exit_ts: dict[str, datetime] = {}

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # reuse for all outputs
    debug_rows: dict[str, list[dict]] = {} if debug else {}
    if debug:
        Path("data/backtests/debug").mkdir(parents=True, exist_ok=True)
        first_signal_seen: set[str] = set()
        first_signal_rows: list[dict] = []

    # ATR trailing stop per symbol (float)
    trail_stop: dict[str, float] = {}

    first_dt = dates[0]
    equity_arr[0] = equity

    # 5) Daily loop
    rng = range(1, len(dates))
    it = _iter_progress(rng, desc=progress_desc) if show_progress else rng
    for i in it:
        is_rebalance_day = (cadence == "daily") or (dates[i].weekday() == 0)
        dt = dates[i]
        prev_dt = dates[i - 1]

        # 5a) Overnight PnL: Close[prev] -> Open[dt] on old holdings (array fast path)
        pnl_overnight = 0.0
        for sym, qty in holdings.items():
            if qty <= 0:
                continue
            sc = sym_cache.get(sym)
            if sc is None:
                continue
            prev_close = sc["close"][i - 1]  # type: ignore[index]
            open_px = sc["open"][i]  # type: ignore[index]
            if np.isfinite(prev_close) and np.isfinite(open_px):
                pnl_overnight += qty * (open_px - prev_close)
        equity += pnl_overnight

        # 5b) Snapshot through prev_dt (signals at D-1, regime veto at D-1)
        if is_rebalance_day:
            todays_prev: dict[str, pd.DataFrame] = {}
            # Use last up-to-200 rows ending at prev index position for rebalancing/sizing
            lo = max(0, i - 199)
            hi = i  # inclusive of prev_dt position
            for sym, sc in sym_cache.items():
                r: pd.DataFrame = sc["frame"]  # type: ignore[assignment]
                snap = r.iloc[lo : hi + 1]
                if "signal" in snap.columns and not regime_arr[i - 1]:
                    # create a tiny copy only when we need to mutate the last row
                    snap = snap.copy()
                    snap.iloc[-1, snap.columns.get_loc("signal")] = 0.0
                todays_prev[sym] = snap

        # Debug: capture the first *eligible* buy signal (after entry_confirm and regime)
        if debug:
            for sym, sc in sym_cache.items():
                if "first_signal_seen" not in locals() or sym in first_signal_seen:
                    continue
                # Require regime ok at prev index
                if not regime_arr[i - 1]:
                    continue
                sig_val = sc["signal"][i - 1]  # type: ignore[index]
                if not np.isfinite(sig_val):
                    continue
                # If entry_confirm > 1, use precomputed boolean
                if entry_confirm and entry_confirm > 1:
                    if not bool(sc["entry_ok"][i - 1]):  # type: ignore[index]
                        continue
                else:
                    if sig_val <= 0:
                        continue

                close_prev = sc["close"][i - 1]  # type: ignore[index]
                atr_prev = sc["atr"][i - 1]  # type: ignore[index]
                don_prev = sc["don"][i - 1]  # type: ignore[index]

                first_signal_rows.append(
                    {
                        "symbol": sym,
                        "date": prev_dt,
                        "close_prev": (
                            float(close_prev) if np.isfinite(close_prev) else float("nan")
                        ),
                        "donchian_upper_prev": (
                            float(don_prev) if np.isfinite(don_prev) else float("nan")
                        ),
                        "atr_prev": float(atr_prev) if np.isfinite(atr_prev) else float("nan"),
                        "signal_prev": float(sig_val),
                        "regime_ok_prev": bool(regime_arr[i - 1]),
                    }
                )
                first_signal_seen.add(sym)

        # 5c) Update/initialize ATR trailing stops with D-1 values; detect breaches
        breached: set[str] = set()
        for sym, qty in holdings.items():
            if qty <= 0:
                continue
            sc = sym_cache.get(sym)
            if sc is None:
                continue
            close_prev = sc["close"][i - 1]  # type: ignore[index]
            atr_prev = sc["atr"][i - 1] if "atr" in sc else np.nan  # type: ignore[index]
            if not np.isfinite(close_prev) or not np.isfinite(atr_prev):
                continue

            # initialize or ratchet
            new_stop = float(close_prev - rules.atr_k * float(atr_prev))
            if sym not in trail_stop or trail_stop[sym] <= 0.0:
                trail_stop[sym] = new_stop
            else:
                if new_stop > trail_stop[sym]:
                    trail_stop[sym] = new_stop

            # breach at D-1 close => exit next open
            if float(close_prev) <= trail_stop[sym]:
                breached.add(sym)

        # 5d) Desired positions from D-1 signals (before enforcing breaches)
        if is_rebalance_day:
            desired_raw = allocate_signals(todays_prev, equity, rules)
        else:
            # no rebalance today: carry current holdings
            desired_raw = {s: holdings.get(s, 0) for s in symbols}

        # 5e) Force exits for breached symbols
        for sym in breached:
            desired_raw[sym] = 0

        # Exit policy: if exit_mode == 'stop', ignore sell signals while in a position.
        # Keep prev position unless a stop was breached; new entries still require buy signals.
        if exit_mode == "stop":
            for sym, prev_qty in list(holdings.items()):
                if prev_qty > 0 and sym not in breached:
                    if desired_raw.get(sym, 0) <= 0:
                        desired_raw[sym] = prev_qty

        # Entry confirmation: require N consecutive buy signals before *new* entries
        if entry_confirm and entry_confirm > 1:
            for sym in list(desired_raw.keys()):
                if holdings.get(sym, 0) == 0 and desired_raw.get(sym, 0) > 0:
                    sc = sym_cache.get(sym)
                    if sc is None:
                        desired_raw[sym] = 0
                        continue
                    if not bool(sc["entry_ok"][i - 1]):  # type: ignore[index]
                        desired_raw[sym] = 0

        # cooldown: block new entries for N bars after an exit
        if cooldown > 0:
            for sym, qty in list(desired_raw.items()):
                # only consider *new* longs (prev 0 -> >0)
                if holdings.get(sym, 0) == 0 and qty > 0:
                    last = last_exit_ts.get(sym)
                    if last is not None:
                        bars_since = (prev_dt - last).days
                        if bars_since < cooldown:
                            desired_raw[sym] = 0

        # 5f) Enforce gross exposure cap at Open[dt] using desired_raw
        gross_notional = 0.0
        for sym, qty in desired_raw.items():
            sc = sym_cache.get(sym)
            if sc is None:
                continue
            px_open = sc["open"][i]  # type: ignore[index]
            if not np.isfinite(px_open):
                continue
            gross_notional += abs(qty) * float(px_open)
        cap_notional = rules.max_gross_exposure_pct * equity
        scale = (
            (cap_notional / gross_notional)
            if (cap_notional > 0 and gross_notional > 0 and gross_notional > cap_notional)
            else 1.0
        )

        desired = {}
        for sym, qty in desired_raw.items():
            if scale < 1.0:
                scaled = int(abs(qty) * scale)
                desired[sym] = scaled if qty >= 0 else -scaled
            else:
                desired[sym] = qty

        # ---- DEBUG ROW CAPTURE (D-1 snapshot driving D open actions) ----
        if debug:
            for sym in symbols:
                sc = sym_cache.get(sym)
                prev_qty = holdings.get(sym, 0)
                new_qty = desired.get(sym, 0)

                if sc is not None:
                    close_prev = sc["close"][i - 1]  # type: ignore[index]
                    atr_prev = sc["atr"][i - 1]  # type: ignore[index]
                    sig_prev = sc["signal"][i - 1]  # type: ignore[index]
                else:
                    close_prev = np.nan
                    atr_prev = np.nan
                    sig_prev = np.nan

                regime_prev = bool(regime_arr[i - 1])

                row = {
                    "date": prev_dt,  # decision basis date (D-1)
                    "close_prev": float(close_prev) if np.isfinite(close_prev) else float("nan"),
                    "atr_prev": float(atr_prev) if np.isfinite(atr_prev) else float("nan"),
                    "trail_stop_prev": float(trail_stop.get(sym, float("nan"))),
                    "signal_prev": float(sig_prev) if np.isfinite(sig_prev) else float("nan"),
                    "prev_qty": int(prev_qty),
                    "new_qty": int(new_qty),
                    "breached_stop": bool(sym in locals().get("breached", set())),
                    "regime_ok_prev": regime_prev,
                }
                debug_rows.setdefault(sym, []).append(row)
        # ---- /DEBUG ROW CAPTURE ----

        # 5f.1) Min-rebalance notional filter
        # Skip tiny rebalances/entries when the notional change is below
        # `min_rebalance_pct * equity`. Always allow full exits.
        if min_rebalance_pct and float(min_rebalance_pct) > 0.0:
            threshold_notional = float(min_rebalance_pct) * float(equity)
            filtered: dict[str, int] = {}
            skipped_small = 0
            skipped_notional = 0.0

            # Evaluate against current open price for today (execution price)
            for sym in set(holdings) | set(desired):
                prev_qty = int(holdings.get(sym, 0))
                new_qty = int(desired.get(sym, 0))

                # No change
                if new_qty == prev_qty:
                    filtered[sym] = new_qty
                    continue

                sc = sym_cache.get(sym)
                if sc is None:
                    filtered[sym] = new_qty
                    continue

                px_open = sc["open"][i]  # type: ignore[index]
                if not np.isfinite(px_open):
                    filtered[sym] = new_qty
                    continue

                delta_qty = new_qty - prev_qty
                delta_notional = abs(delta_qty) * float(px_open)

                # Always allow closes (including ATR stop or signal exits)
                force_close = prev_qty > 0 and new_qty == 0

                if (not force_close) and (delta_notional < threshold_notional):
                    # Skip tiny change: keep previous position
                    filtered[sym] = prev_qty
                    skipped_small += 1
                    skipped_notional += delta_notional
                else:
                    filtered[sym] = new_qty

            # Replace desired with filtered positions
            desired = filtered
            if skipped_small and not quiet_mode:
                logger.debug(
                    f"Min-rebalance filter skipped {skipped_small} small order(s) (~${skipped_notional:,.2f} notional) on {dt.date()}"
                )

        # 5g) Compute transaction costs at Open[dt] for quantity deltas
        cost = 0.0
        for sym in set(holdings) | set(desired):
            sc = sym_cache.get(sym)
            if sc is None:
                continue
            px_open = sc["open"][i]  # type: ignore[index]
            if not np.isfinite(px_open):
                continue
            delta = abs(desired.get(sym, 0) - holdings.get(sym, 0))
            if delta > 0:
                cost += (rules.cost_bps / 10_000.0) * float(px_open) * delta
        if cost:
            total_costs += cost
            equity -= cost

        # 5h) Post synthetic market fills at Open[dt] into the ledger
        all_syms = set(holdings) | set(desired)
        for sym in all_syms:
            sc = sym_cache.get(sym)
            if sc is None:
                continue
            px_open = sc["open"][i]  # type: ignore[index]
            if not np.isfinite(px_open):
                continue
            prev_qty = holdings.get(sym, 0)
            new_qty = desired.get(sym, 0)
            if new_qty == prev_qty:
                continue

            if prev_qty > 0 and new_qty == 0 and sym in locals().get("breached", set()):
                reason = "atr_trail_stop"
            else:
                reason = "rebalance"

            ledger.submit_and_fill(
                symbol=sym,
                new_qty=new_qty,
                price=px_open,
                ts=dt.to_pydatetime(),
                reason=reason,
                cost_usd=0.0,  # cost already deducted globally
            )

            # initialize trailing stop on first entry (using D-1)
            if prev_qty == 0 and new_qty > 0:
                sc2 = sym_cache.get(sym)
                if sc2 is not None:
                    close_prev = sc2["close"][i - 1]  # type: ignore[index]
                    atr_prev = sc2["atr"][i - 1]  # type: ignore[index]
                    if np.isfinite(close_prev) and np.isfinite(atr_prev):
                        trail_stop[sym] = float(close_prev - rules.atr_k * float(atr_prev))

        # 5i) Commit new holdings
        holdings = desired

        # 5j) Intraday PnL: Open[dt] -> Close[dt] on new holdings (array fast path)
        pnl_intraday = 0.0
        for sym, qty in holdings.items():
            if qty == 0:
                continue
            sc = sym_cache.get(sym)
            if sc is None:
                continue
            open_px = sc["open"][i]  # type: ignore[index]
            close_px = sc["close"][i]  # type: ignore[index]
            if np.isfinite(open_px) and np.isfinite(close_px):
                pnl_intraday += qty * (float(close_px) - float(open_px))
        equity += pnl_intraday

        equity_arr[i] = equity

    # 6) Outputs
    os.makedirs(outdir, exist_ok=True)
    equity_series = pd.Series(equity_arr, index=dates_idx, name="equity")

    # Optional: write first-signal snapshots if collected
    if debug:
        dbg_dir = Path("data/backtests/debug")
        dbg_dir.mkdir(parents=True, exist_ok=True)
        if "first_signal_rows" in locals() and first_signal_rows:
            fs_path = dbg_dir / f"PORT_{strategy.name}_{stamp}_first_signals.csv"
            pd.DataFrame(first_signal_rows).to_csv(fs_path, index=False)
            logger.info(f"Debug: first-signal snapshots saved to {fs_path}")

    port_outfile = os.path.join(outdir, f"PORT_{strategy.name}_{stamp}.csv")
    plot_path = os.path.join(outdir, f"PORT_{strategy.name}_{stamp}.png")
    summary_path = os.path.join(outdir, f"PORT_{strategy.name}_{stamp}_summary.csv")
    trades_path = os.path.join(outdir, f"PORT_{strategy.name}_{stamp}_trades.csv")

    # Write equity curve CSV (optional)
    if write_portfolio_csv:
        equity_series.to_frame("equity").to_csv(port_outfile)

    # Summary dict + optional write
    m = perf_metrics(equity_series.dropna())
    summary: dict[str, object] = {
        **m,
        "total_costs": float(total_costs),
        "rules_per_trade_risk_pct": float(rules.per_trade_risk_pct),
        "rules_atr_k": float(rules.atr_k),
        "rules_max_positions": float(rules.max_positions),
        "rules_cost_bps": float(rules.cost_bps),
        "rules_max_gross_exposure_pct": float(rules.max_gross_exposure_pct),
        "regime_on": bool(regime_on),
        "regime_symbol": regime_symbol,
        "regime_window": float(regime_window),
    }
    if write_summary_csv:
        pd.Series(summary).to_csv(summary_path, header=False)

    # Plot only if explicitly requested
    if make_plot:
        plt.figure()
        equity_series.plot(title=f"Equity Curve - {strategy.name}")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

    # Trades export from ledger (optional)
    trades_df = None
    if write_trades_csv:
        trades_df = ledger.to_trades_dataframe()
        trades_df.to_csv(trades_path, index=False)
    elif return_trades:
        # materialize for in-memory consumption only
        trades_df = ledger.to_trades_dataframe()

    # Logging reflects what we actually wrote
    if not quiet_mode:
        if write_portfolio_csv and make_plot:
            logger.info(f"Portfolio results saved to {port_outfile} and {plot_path}")
        elif write_portfolio_csv and not make_plot:
            logger.info(f"Portfolio results saved to {port_outfile} (plot skipped)")
        elif (not write_portfolio_csv) and make_plot:
            logger.info(f"Portfolio plot saved to {plot_path}")
        else:
            logger.info("Portfolio outputs skipped (no CSV/plot)")

        if write_summary_csv:
            logger.info(f"Summary saved to {summary_path}")
        else:
            logger.info("Summary CSV skipped (in-memory only)")

        if write_trades_csv:
            logger.info(f"Trades saved to {trades_path}")
        else:
            logger.info("Trades CSV skipped")

    # Optional in-memory return for sweep/CLI fast-path (avoids CSV round-trips)
    if return_summary or return_equity or return_trades:
        result: dict[str, object] = {"summary": summary}
        if return_equity:
            result["equity"] = equity_series
        if return_trades and trades_df is not None:
            result["trades"] = trades_df
        return result

    if not quiet_mode:
        logger.info(
            "Total: {total:.2%} | CAGR~: {cagr:.2%} | Sharpe: {sharpe:.2f} | "
            "MaxDD: {mdd:.2%} | Costs: ${costs:,.2f}".format(
                total=m["total_return"],
                cagr=m["cagr"],
                sharpe=m["sharpe"],
                mdd=m["max_drawdown"],
                costs=total_costs,
            )
        )
        logger.info("Backtest complete.")


def validate_indicators(df: pd.DataFrame, symbol: str, atr_col: str = "ATR") -> None:
    """
    Soft validation for indicators. If ATR is present, ensure all positive (no zeros/negatives)
    to avoid division-by-zero or nonsensical sizing.
    Raises ValueError if ATR exists and has non-positive values.
    """
    if atr_col in df.columns:
        # Consider most recent ATR values; if any nonpositive present, raise
        atr_vals = df[atr_col].astype(float)
        if (atr_vals <= 0).any():
            raise ValueError(f"{symbol}: non-positive ATR detected")

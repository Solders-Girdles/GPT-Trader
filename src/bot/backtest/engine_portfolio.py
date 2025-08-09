from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bot.dataflow.sources.yfinance_source import YFinanceSource
from bot.dataflow.validate import adjust_to_adjclose, validate_daily_bars
from bot.exec.ledger import Ledger
from bot.logging import get_logger
from bot.metrics.report import perf_metrics
from bot.portfolio.allocator import PortfolioRules, allocate_signals
from bot.strategy.base import Strategy

logger = get_logger("backtest")


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
    min_rebalance_pct: float = 0.0,
    strict_mode: bool = True,
) -> None:
    """
    v1.5 timing model + ATR trailing exits:
    - Signals use data through D-1.
    - Trades execute at D open (price + costs at Open[D]).
    - PnL = overnight (Close[D-1]→Open[D]) on old holdings
      + intraday (Open[D]→Close[D]) on new holdings.
    - Regime filter (SPY > 200DMA) evaluated at D-1 (strict SMA).
    - ATR trailing stop: computed from D-1 Close/ATR, ratchets up only.
      If breached at D-1, exit at D open.
    """
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

    # 1) Load data
    data_map: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = src.get_daily_bars(
                sym,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
            # --- Data adjustments & validation ---
            df_adj, did_adj = adjust_to_adjclose(df)
            # Validate after adjustment to catch gaps/NaNs/bounds
            df_adj = validate_ohlc(df_adj, sym, strict_mode=strict_mode)
            if did_adj:
                logger.info(f"{sym}: applied OHLC adjustment using Adj Close factors")
            validate_daily_bars(df_adj, sym)
            df = df_adj
            data_map[sym] = df
        except Exception as e:
            logger.warning(f"Skipping {sym}: {e}")

    if not data_map:
        logger.error("No data loaded; aborting.")
        return

    # 2) Regime series (optional)
    regime_ok: pd.Series | None = None
    if regime_on:
        try:
            mkt = src.get_daily_bars(
                regime_symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
            ma = _strict_sma(mkt["Close"], regime_window)
            regime_ok = (mkt["Close"] > ma).astype(bool)
        except Exception as e:
            logger.warning(f"Regime fetch failed ({regime_symbol}): {e}")
            regime_ok = None

    # 3) Signals per symbol
    sigs: dict[str, pd.DataFrame] = {}
    for sym, df in data_map.items():
        s = strategy.generate_signals(df)
        sigs[sym] = df.join(s, how="left")

    dates = sorted(set().union(*[d.index for d in data_map.values()]))
    if len(dates) < 2:
        logger.error("Not enough dates to run a next-open backtest.")
        return

    # 4) State
    equity0 = 100_000.0
    equity_series = pd.Series(index=pd.Index(dates, name="Date"), dtype=float)
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
    # ATR trailing stop per symbol (float)
    trail_stop: dict[str, float] = {}

    first_dt = dates[0]
    equity_series.loc[first_dt] = equity

    # 5) Daily loop
    for i in range(1, len(dates)):
        is_rebalance_day = (cadence == "daily") or (dates[i].weekday() == 0)
        dt = dates[i]
        prev_dt = dates[i - 1]

        # 5a) Overnight PnL: Close[prev] -> Open[dt] on old holdings
        pnl_overnight = 0.0
        for sym, qty in holdings.items():
            df = data_map.get(sym)
            if df is None or prev_dt not in df.index or dt not in df.index:
                continue
            prev_close = _safe_float(df.loc[prev_dt, "Close"])
            # Execution & sizing use Open[dt] (next open) to avoid Close→Open mismatch.
            open_px = _safe_float(df.loc[dt, "Open"])
            pnl_overnight += qty * (open_px - prev_close)
        equity += pnl_overnight

        # 5b) Snapshot through prev_dt (signals at D-1, regime veto at D-1)
        todays_prev: dict[str, pd.DataFrame] = {}
        for sym, df in sigs.items():
            if prev_dt in df.index:
                snap = df.loc[:prev_dt].tail(200).copy()
                if (
                    regime_ok is not None
                    and prev_dt in regime_ok.index
                    and not bool(regime_ok.loc[prev_dt])
                    and "signal" in snap.columns
                ):
                    snap.iloc[-1, snap.columns.get_loc("signal")] = 0
                todays_prev[sym] = snap

        # 5c) Update/initialize ATR trailing stops with D-1 values; detect breaches
        breached: set[str] = set()
        for sym, qty in holdings.items():
            if qty <= 0:
                continue
            snap = todays_prev.get(sym)
            if snap is None or snap.empty:
                continue
            close_prev = float(snap["Close"].iloc[-1])
            atr_prev = float(snap["atr"].iloc[-1]) if "atr" in snap.columns else 0.0

            # initialize if missing
            if sym not in trail_stop or trail_stop[sym] <= 0.0:
                trail_stop[sym] = close_prev - rules.atr_k * atr_prev
            else:
                # ratchet up only
                new_stop = close_prev - rules.atr_k * atr_prev
                if new_stop > trail_stop[sym]:
                    trail_stop[sym] = new_stop

            # breach at D-1 close => exit next open
            if close_prev <= trail_stop[sym]:
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
        if entry_confirm > 1:
            for sym in list(desired_raw.keys()):
                prev_q = holdings.get(sym, 0)
                if prev_q == 0 and desired_raw.get(sym, 0) > 0:
                    snap = todays_prev.get(sym)
                    ok = False
                    if snap is not None and not snap.empty and "signal" in snap.columns:
                        # look at last N rows (<= prev_dt) and require strictly positive signals
                        lastN = snap["signal"].tail(entry_confirm)
                        ok = (len(lastN) == entry_confirm) and (lastN > 0).all()
                    if not ok:
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

        # 5f) Enforce gross exposure cap at Open[dt]
        gross_notional = 0.0
        for sym, qty in desired_raw.items():
            df = data_map.get(sym)
            if df is None or dt not in df.index:
                continue
            px_open = _safe_float(df.loc[dt, "Open"])
            # skip micro rebalances by notional threshold
            prev_qty = holdings.get(sym, 0)
            new_qty = desired.get(sym, 0)
            if new_qty == prev_qty:
                continue
            delta = abs(new_qty - prev_qty)
            notional_change = delta * px_open
            if min_rebalance_pct > 0.0 and notional_change < (min_rebalance_pct * equity):
                continue
            delta = abs(new_qty - prev_qty)
            notional_change = delta * px_open
            gross_notional += abs(qty) * px_open
        cap_notional = rules.max_gross_exposure_pct * equity
        if cap_notional > 0 and gross_notional > cap_notional:
            scale = cap_notional / gross_notional
        else:
            scale = 1.0

        desired = {}
        if scale < 1.0:
            for sym, qty in desired_raw.items():
                scaled = int(abs(qty) * scale)
                desired[sym] = scaled if qty >= 0 else -scaled
        else:
            desired = desired_raw

        # ---- DEBUG ROW CAPTURE (D-1 snapshot driving D open actions) ----
        if debug:
            for sym in symbols:
                prev_qty = holdings.get(sym, 0)
                new_qty = desired.get(sym, 0)

                snap = todays_prev.get(sym)
                if snap is not None and not snap.empty:
                    close_prev = float(snap["Close"].iloc[-1])
                    atr_prev = (
                        float(snap["atr"].iloc[-1]) if "atr" in snap.columns else float("nan")
                    )
                    sig_prev = (
                        float(snap["signal"].iloc[-1]) if "signal" in snap.columns else float("nan")
                    )
                else:
                    close_prev = float("nan")
                    atr_prev = float("nan")
                    sig_prev = float("nan")

                regime_prev = (
                    bool(regime_ok.loc[prev_dt])
                    if (regime_ok is not None and prev_dt in regime_ok.index)
                    else True
                )

                # if the ATR trailing set flagged a breach earlier this loop
                breached_flag = sym in locals().get("breached", set())

                row = {
                    "date": prev_dt,  # decision basis date (D-1)
                    "close_prev": close_prev,
                    "atr_prev": atr_prev,
                    "trail_stop_prev": float(trail_stop.get(sym, float("nan"))),
                    "signal_prev": sig_prev,
                    "prev_qty": int(prev_qty),
                    "new_qty": int(new_qty),
                    "breached_stop": bool(breached_flag),
                    "regime_ok_prev": bool(regime_prev),
                }
                debug_rows.setdefault(sym, []).append(row)
        # ---- /DEBUG ROW CAPTURE ----

        # 5g) Compute transaction costs at Open[dt] for quantity deltas
        cost = 0.0
        all_syms = set(holdings) | set(desired)
        for sym in all_syms:
            # skip micro rebalances
            # compute notional change at open
            df = data_map.get(sym)
            if df is None or dt not in df.index:
                continue
            px_open = _safe_float(df.loc[dt, "Open"])
            prev_qty = holdings.get(sym, 0)
            new_qty = desired.get(sym, 0)
            delta = abs(new_qty - prev_qty)
            notional_change = delta * px_open

            # (cost computation continues below)
            prev_qty = holdings.get(sym, 0)
            new_qty = desired.get(sym, 0)
            delta = abs(new_qty - prev_qty)
            if delta <= 0:
                continue
            df = data_map.get(sym)
            if df is None or dt not in df.index:
                continue
            px_open = _safe_float(df.loc[dt, "Open"])
            cost += (rules.cost_bps / 10_000.0) * px_open * delta
        if cost:
            total_costs += cost
            equity -= cost  # deduct globally

        # 5h) Post synthetic market fills at Open[dt] into the ledger
        for sym in all_syms:
            df = data_map.get(sym)
            if df is None or dt not in df.index:
                continue
            px_open = _safe_float(df.loc[dt, "Open"])
            prev_qty = holdings.get(sym, 0)
            new_qty = desired.get(sym, 0)
            if new_qty == prev_qty:
                continue

            if prev_qty > 0 and new_qty == 0 and sym in breached:
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
                snap = todays_prev.get(sym)
                if snap is not None and not snap.empty:
                    close_prev = float(snap["Close"].iloc[-1])
                    atr_prev = float(snap["atr"].iloc[-1]) if "atr" in snap.columns else 0.0
                    trail_stop[sym] = close_prev - rules.atr_k * atr_prev

        # 5i) Commit new holdings
        holdings = desired

        # 5j) Intraday PnL: Open[dt] -> Close[dt] on new holdings
        pnl_intraday = 0.0
        for sym, qty in holdings.items():
            df = data_map.get(sym)
            if df is None or dt not in df.index:
                continue
            open_px = _safe_float(df.loc[dt, "Open"])
            close_px = _safe_float(df.loc[dt, "Close"])
            pnl_intraday += qty * (close_px - open_px)
        equity += pnl_intraday

        equity_series.loc[dt] = equity

    # 6) Outputs
    os.makedirs(outdir, exist_ok=True)

    port_outfile = os.path.join(outdir, f"PORT_{strategy.name}_{stamp}.csv")
    equity_series.to_frame("equity").to_csv(port_outfile)

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

    plt.figure()
    equity_series.plot(title=f"Equity Curve - {strategy.name}")
    plot_path = os.path.join(outdir, f"PORT_{strategy.name}_{stamp}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    summary_path = os.path.join(outdir, f"PORT_{strategy.name}_{stamp}_summary.csv")
    pd.Series(summary).to_csv(summary_path, header=False)

    # Trades export from ledger
    trades_df = ledger.to_trades_dataframe()
    trades_path = os.path.join(outdir, f"PORT_{strategy.name}_{stamp}_trades.csv")
    trades_df.to_csv(trades_path, index=False)

    logger.info(f"Portfolio results saved to {port_outfile} and {plot_path}")
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"Trades saved to {trades_path}")
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

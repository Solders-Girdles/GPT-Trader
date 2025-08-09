from __future__ import annotations

import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from bot.dataflow.sources.yfinance_source import YFinanceSource
from bot.logging import get_logger
from bot.metrics.report import perf_metrics
from bot.portfolio.allocator import PortfolioRules, allocate_signals
from bot.strategy.base import Strategy

logger = get_logger("backtest")


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
    # Robustly extract a float from pandas scalars/1-elt Series or plain numbers.
    if hasattr(x, "item"):
        return float(x.item())  # pandas scalar
    if hasattr(x, "iloc"):
        return float(x.iloc[0])  # 1-element Series
    return float(x)


def _strict_sma(series: pd.Series, window: int) -> pd.Series:
    # SMA that is only "valid" after full lookback (no early start).
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
) -> None:
    """
    v1.5 timing model:
    - Signals are computed on data up to D-1 (prev close).
    - Trades execute at D open (price + costs at Open[D]).
    - PnL includes (a) overnight prev Close -> Open[D] on prior holdings,
      then (b) intraday Open[D] -> Close[D] on new holdings.
    - Strict regime filter (SPY > 200DMA with min_periods=window) is evaluated at D-1.
    """
    logger.info(f"Backtesting {strategy.name} from {start.date()} to {end.date()}")
    outdir = os.path.join("data", "backtests")
    os.makedirs(outdir, exist_ok=True)

    src = YFinanceSource()

    if symbol_list_csv:
        symbols = _read_universe_csv(symbol_list_csv)
    elif symbol:
        symbols = [symbol]
    else:
        raise ValueError("Provide --symbol or --symbol-list")

    # Load data for all symbols (must include Open/High/Low/Close/Volume)
    data_map: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = src.get_daily_bars(
                sym,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
            # Ensure Open exists; if not, fallback to Close for open/close ops
            if "Open" not in df.columns:
                df["Open"] = df["Close"]
            data_map[sym] = df
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Skipping {sym}: {e}")

    if not data_map:
        logger.error("No data loaded; aborting.")
        return

    # Regime (SPY > 200DMA) if enabled — strict (valid only after full lookback)
    # We'll consult regime at D-1 when deciding trades for D.
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
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Regime fetch failed ({regime_symbol}): {e}")
            regime_ok = None

    # Signals/indicators per symbol (join onto OHLCV)
    sigs: dict[str, pd.DataFrame] = {}
    for sym, df in data_map.items():
        s = strategy.generate_signals(df)
        sigs[sym] = df.join(s, how="left")

    # Trading calendar
    dates = sorted(set().union(*[d.index for d in data_map.values()]))
    if len(dates) < 2:
        logger.error("Not enough dates to run a next-open backtest.")
        return

    equity0 = 100_000.0
    equity_series = pd.Series(index=pd.Index(dates, name="Date"), dtype=float)
    equity = equity0
    holdings: dict[str, int] = {}
    total_costs = 0.0

    # initialize first day (no trades at first date)
    first_dt = dates[0]
    equity_series.loc[first_dt] = equity

    # Main loop starts at second date to allow next-open execution
    for i in range(1, len(dates)):
        dt = dates[i]
        prev_dt = dates[i - 1]

        # 1) Overnight PnL on existing holdings: Close[prev] -> Open[dt]
        pnl_overnight = 0.0
        for sym, qty in holdings.items():
            df = data_map.get(sym)
            if df is None or prev_dt not in df.index or dt not in df.index:
                continue
            prev_close = _safe_float(df.loc[prev_dt, "Close"])
            open_px = _safe_float(df.loc[dt, "Open"])
            pnl_overnight += qty * (open_px - prev_close)
        equity += pnl_overnight

        # 2) Build per-symbol snapshot up to prev_dt (signals at D-1)
        todays_prev: dict[str, pd.DataFrame] = {}
        for sym, df in sigs.items():
            if prev_dt in df.index:
                snap = df.loc[:prev_dt].tail(200).copy()
                # Regime veto at D-1 (zero the last signal if regime says off)
                if (
                    regime_ok is not None
                    and prev_dt in regime_ok.index
                    and not bool(regime_ok.loc[prev_dt])
                    and "signal" in snap.columns
                ):
                    snap.iloc[-1, snap.columns.get_loc("signal")] = 0
                todays_prev[sym] = snap

        # 3) Compute desired positions from D-1 signals using current equity
        desired_raw = allocate_signals(todays_prev, equity, rules)

        # 4) Enforce exact gross exposure cap at Open[dt]
        #    Scale quantities proportionally if cap exceeded.
        gross_notional = 0.0
        for sym, qty in desired_raw.items():
            df = data_map.get(sym)
            if df is None or dt not in df.index:
                continue
            px_open = _safe_float(df.loc[dt, "Open"])
            gross_notional += abs(qty) * px_open
        cap_notional = rules.max_gross_exposure_pct * equity
        scale = 1.0
        if cap_notional > 0 and gross_notional > cap_notional:
            scale = cap_notional / gross_notional

        desired: dict[str, int] = {}
        if scale < 1.0:
            for sym, qty in desired_raw.items():
                scaled = int(abs(qty) * scale)
                desired[sym] = scaled if qty >= 0 else -scaled
        else:
            desired = desired_raw

        # 5) Transaction costs on quantity changes at Open[dt], then set holdings
        #    (In v1.5, we assume instant fills at Open[dt].)
        cost = 0.0
        all_syms = set(holdings) | set(desired)
        for sym in all_syms:
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
            equity -= cost

        holdings = desired

        # 6) Intraday PnL on new holdings: Open[dt] -> Close[dt]
        pnl_intraday = 0.0
        for sym, qty in holdings.items():
            df = data_map.get(sym)
            if df is None or dt not in df.index:
                continue
            open_px = _safe_float(df.loc[dt, "Open"])
            close_px = _safe_float(df.loc[dt, "Close"])
            pnl_intraday += qty * (close_px - open_px)
        equity += pnl_intraday

        # 7) Record equity
        equity_series.loc[dt] = equity

    # Outputs
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join("data", "backtests")
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

    # Plot equity curve
    plt.figure()
    equity_series.plot(title=f"Equity Curve - {strategy.name}")
    plot_path = os.path.join(outdir, f"PORT_{strategy.name}_{stamp}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    # Summary CSV
    summary_path = os.path.join(outdir, f"PORT_{strategy.name}_{stamp}_summary.csv")
    pd.Series(summary).to_csv(summary_path, header=False)

    logger.info(f"Portfolio results saved to {port_outfile} and {plot_path}")
    logger.info(f"Summary saved to {summary_path}")
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

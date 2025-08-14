from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from bot.validation import get_math_validator


@dataclass
class PortfolioRules:
    per_trade_risk_pct: float = 0.005  # 0.5% of equity risk per trade
    atr_k: float = 2.0
    max_positions: int = 10
    max_gross_exposure_pct: float = 0.60
    cost_bps: float = 0.0  # one-way transaction cost (bps)
    # Phase 2: execution cost hooks and turnover guard
    cost_adjusted_sizing: bool = False  # if true, reduce risk by expected costs
    slippage_bps: float = 0.0  # estimated one-way slippage (bps)
    max_turnover_per_rebalance: float | None = None  # optional turnover cap at execution (0-1)


def _to_float(x: pd.Series | float | int) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    if hasattr(x, "iloc"):
        return float(x.iloc[0])
    return float(x)


def position_size(equity: float, atr_value: float, price: float, rules: PortfolioRules) -> int:
    risk_usd = equity * rules.per_trade_risk_pct
    if rules.cost_adjusted_sizing:
        # Reduce available risk by expected entry cost and slippage (conservative)
        total_cost_rate = max(0.0, (rules.cost_bps + rules.slippage_bps) / 10_000.0)
        risk_usd = max(0.0, risk_usd * (1.0 - total_cost_rate))
    stop_dist = rules.atr_k * atr_value
    if stop_dist <= 0 or price <= 0:
        return 0
    qty = math.floor(risk_usd / stop_dist)
    return max(qty, 0)


def allocate_signals(
    signals: dict[str, pd.DataFrame], equity: float, rules: PortfolioRules
) -> dict[str, int]:
    """
    Rank candidates by breakout strength and pick top N. Returns desired quantities.
    Expects columns: Close, donchian_upper, atr, signal.
    """
    candidates: list[tuple[str, float, int]] = []
    for sym, df in signals.items():
        df = df.dropna()
        if df.empty:
            continue

        # must have a long signal on the latest row
        sig = df["signal"].iloc[-1] if "signal" in df.columns else 0
        if int(_to_float(sig)) <= 0:
            continue

        price = _to_float(df["Open"].iloc[-1] if "Open" in df.columns else df["Close"].iloc[-1])
        atr_val = _to_float(df.get("atr", pd.Series([0.0])).iloc[-1])
        qty = position_size(equity, atr_val, price, rules)
        if qty <= 0:
            continue

        # Donchian-aware strength if available; otherwise a generic signal/ATR score.

        if "donchian_upper" in df.columns:
            upper_prev = df["donchian_upper"].shift(1).dropna()

            if upper_prev.empty:
                continue

            strength = (price - float(upper_prev.iloc[-1])) / max(price, 1e-9)

        else:
            # Generic strength: prefer higher signal and lower ATR.

            # Avoid div-by-zero; if ATR missing, treat as 1.

            atr_for_rank = float(df["atr"].iloc[-1]) if "atr" in df.columns else 1.0

            # If signal missing, treat as 0 (won't get here because we gate on sig>0 above).

            sig_val = float(df["signal"].iloc[-1]) if "signal" in df.columns else 0.0

            # Use safe division to prevent division by zero
            math_validator = get_math_validator()
            strength = math_validator.safe_divide(
                sig_val, atr_for_rank, default=0.0, name=f"signal_strength_{sym}"
            )

        candidates.append((sym, strength, qty))

    candidates.sort(key=lambda x: x[1], reverse=True)
    picks = candidates[: rules.max_positions]
    return {sym: qty for sym, _, qty in picks}

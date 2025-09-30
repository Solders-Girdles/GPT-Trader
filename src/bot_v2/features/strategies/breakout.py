from __future__ import annotations

from typing import Any

from bot_v2.features.strategies.interfaces import StrategyBase, StrategyContext, StrategySignal


class BreakoutStrategy(StrategyBase):
    name = "breakout"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.breakout_period = int(params.get("breakout_period", 20))
        self.threshold_pct = float(params.get("threshold_pct", 0.01))

    def get_signals(self, ctx: StrategyContext) -> list[StrategySignal]:
        out: list[StrategySignal] = []
        for sym in ctx.symbols:
            prices = self.price_history.get(sym) or []
            if len(prices) < self.breakout_period:
                continue
            # exclude current price when computing recent bands
            window = prices[-self.breakout_period : -1]
            if not window:
                continue
            recent_high = max(window)
            recent_low = min(window)
            curr = prices[-1]
            if curr > recent_high * (1 + self.threshold_pct):
                out.append(StrategySignal(symbol=sym, side="buy", confidence=1.0))
            elif curr < recent_low * (1 - self.threshold_pct):
                out.append(StrategySignal(symbol=sym, side="sell", confidence=1.0))
        return out

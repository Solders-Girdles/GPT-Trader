from __future__ import annotations

from statistics import mean
from typing import List

from .interfaces import StrategyBase, StrategyContext, StrategySignal


class MAStrategy(StrategyBase):
    name = "ma_crossover"

    def __init__(self, **params):
        super().__init__(**params)
        self.fast_period = int(params.get("fast_period", 10))
        self.slow_period = int(params.get("slow_period", 30))
        self.lookback = max(self.fast_period, self.slow_period) + 2

    def get_signals(self, ctx: StrategyContext) -> List[StrategySignal]:
        out: List[StrategySignal] = []
        for sym in ctx.symbols:
            prices = self.price_history.get(sym) or []
            if len(prices) < self.slow_period + 1:
                continue
            fast = mean(prices[-self.fast_period :])
            slow = mean(prices[-self.slow_period :])
            prev_fast = mean(prices[-self.fast_period - 1 : -1])
            prev_slow = mean(prices[-self.slow_period - 1 : -1])
            if prev_fast <= prev_slow and fast > slow:
                out.append(StrategySignal(symbol=sym, side="buy", confidence=1.0))
            elif prev_fast >= prev_slow and fast < slow:
                out.append(StrategySignal(symbol=sym, side="sell", confidence=1.0))
        return out


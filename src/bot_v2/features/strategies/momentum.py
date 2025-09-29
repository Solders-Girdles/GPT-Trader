from __future__ import annotations

from typing import Any

from .interfaces import StrategyBase, StrategyContext, StrategySignal


class MomentumStrategy(StrategyBase):
    name = "momentum"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.momentum_period = int(params.get("momentum_period", 10))
        self.threshold = float(params.get("threshold", 0.02))

    def get_signals(self, ctx: StrategyContext) -> list[StrategySignal]:
        out: list[StrategySignal] = []
        for sym in ctx.symbols:
            prices = self.price_history.get(sym) or []
            if len(prices) <= self.momentum_period:
                continue
            old = prices[-self.momentum_period - 1]
            curr = prices[-1]
            if old == 0:
                continue
            momentum = (curr - old) / old
            if momentum > self.threshold:
                out.append(
                    StrategySignal(
                        symbol=sym, side="buy", confidence=min(1.0, momentum / self.threshold)
                    )
                )
            elif momentum < -self.threshold:
                out.append(
                    StrategySignal(
                        symbol=sym, side="sell", confidence=min(1.0, abs(momentum) / self.threshold)
                    )
                )
        return out

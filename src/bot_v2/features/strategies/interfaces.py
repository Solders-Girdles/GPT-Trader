from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Optional


@dataclass
class StrategyContext:
    symbols: List[str]
    # Add room for future context: balances, positions, market regime, etc.


@dataclass
class StrategySignal:
    symbol: str
    side: str  # 'buy' | 'sell' | 'hold'
    confidence: float = 1.0  # 0..1 for sizing


class IStrategy(Protocol):
    name: str

    def update_price(self, symbol: str, price: float) -> None:
        ...

    def get_signals(self, ctx: StrategyContext) -> List[StrategySignal]:
        ...


class StrategyBase:
    name: str = "base"

    def __init__(self, **params):
        self.params = params
        self.price_history: Dict[str, List[float]] = {}

    def update_price(self, symbol: str, price: float) -> None:
        history = self.price_history.setdefault(symbol, [])
        history.append(price)
        # cap to a reasonable length to avoid unbounded growth
        if len(history) > int(self.params.get("lookback", 500)):
            del history[: len(history) - int(self.params.get("lookback", 500))]

    def get_signals(self, ctx: StrategyContext) -> List[StrategySignal]:
        return []


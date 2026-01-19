"""Shared helpers for hybrid strategy tests."""

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.hybrid.base import HybridStrategyBase
from gpt_trader.features.live_trade.strategies.hybrid.types import (
    HybridDecision,
    HybridMarketData,
    HybridPositionState,
    HybridStrategyConfig,
)


class ConcreteHybridStrategy(HybridStrategyBase):
    """Concrete implementation for testing."""

    def __init__(self, config: HybridStrategyConfig, decisions: list[HybridDecision] | None = None):
        super().__init__(config)
        self._decisions = decisions or []

    def decide_hybrid(
        self,
        market_data: HybridMarketData,
        positions: HybridPositionState,
        equity: Decimal,
    ) -> list[HybridDecision]:
        return self._decisions

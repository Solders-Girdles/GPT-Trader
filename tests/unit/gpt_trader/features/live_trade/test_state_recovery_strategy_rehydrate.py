"""Tests for strategy-level rehydration hooks."""

from __future__ import annotations

from gpt_trader.features.live_trade.engines.strategy import EVENT_PRICE_TICK
from gpt_trader.features.live_trade.strategies.perps_baseline import BaselinePerpsStrategy


class TestStrategyRehydrate:
    """Tests for strategy rehydrate method."""

    def test_baseline_strategy_rehydrate_returns_zero(self) -> None:
        """Test that BaselinePerpsStrategy.rehydrate returns 0 (stateless)."""
        strategy = BaselinePerpsStrategy()

        events = [
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "BTC-PERP", "price": "50000.00"}},
        ]
        result = strategy.rehydrate(events)

        assert result == 0

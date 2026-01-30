"""Unit tests for StatefulStrategy."""

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from gpt_trader.core import Action, Decision, Product
from gpt_trader.features.live_trade.strategies.stateful import StatefulStrategy


class ConcreteStrategy(StatefulStrategy):
    """Concrete implementation for testing."""

    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: Sequence[Decimal],
        equity: Decimal,
        product: Product | None,
    ) -> Decision:
        self.update_stats(symbol, current_mark)
        return Decision(action=Action.HOLD, reason="test")


class TestStatefulStrategy:
    def test_initialization(self):
        strategy = ConcreteStrategy()
        assert len(strategy.stats) == 0
        assert not strategy._initialized

    def test_update_stats(self):
        strategy = ConcreteStrategy()
        symbol = "BTC-USD"

        strategy.update_stats(symbol, Decimal("100"))
        strategy.update_stats(symbol, Decimal("200"))

        stats = strategy.stats[symbol]
        assert stats.count == 2
        assert stats.mean == 150.0
        assert stats.min_val == 100.0
        assert stats.max_val == 200.0

    def test_rehydrate(self):
        strategy = ConcreteStrategy()
        events = [
            {"type": "price_tick", "data": {"symbol": "BTC-USD", "price": "100"}},
            {"type": "other_event", "data": {}},
            {"type": "price_tick", "data": {"symbol": "BTC-USD", "price": "200"}},
            {"type": "price_tick", "data": {"symbol": "ETH-USD", "price": "1000"}},
        ]

        processed = strategy.rehydrate(events)

        assert processed == 3
        assert strategy._initialized

        btc_stats = strategy.stats["BTC-USD"]
        assert btc_stats.count == 2
        assert btc_stats.mean == 150.0

        eth_stats = strategy.stats["ETH-USD"]
        assert eth_stats.count == 1
        assert eth_stats.mean == 1000.0

    def test_decide_updates_stats(self):
        strategy = ConcreteStrategy()
        symbol = "BTC-USD"

        strategy.decide(
            symbol=symbol,
            current_mark=Decimal("100"),
            position_state=None,
            recent_marks=[],
            equity=Decimal("1000"),
            product=None,
        )

        assert strategy.stats[symbol].count == 1
        assert strategy.stats[symbol].mean == 100.0

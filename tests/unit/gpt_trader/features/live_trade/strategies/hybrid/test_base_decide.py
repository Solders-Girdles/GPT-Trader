"""Tests for HybridStrategyBase.decide()."""

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.hybrid.types import (
    Action,
    HybridDecision,
    HybridStrategyConfig,
    TradingMode,
)
from tests.unit.gpt_trader.features.live_trade.strategies.hybrid.hybrid_strategy_test_helpers import (
    ConcreteHybridStrategy,
)


class TestHybridStrategyBaseDecide:
    """Tests for decide() method (standard interface)."""

    def test_decide_hold_when_no_decisions(self):
        """Returns HOLD when decide_hybrid returns empty list."""
        config = HybridStrategyConfig()
        strategy = ConcreteHybridStrategy(config, decisions=[])

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[],
            equity=Decimal("10000"),
            product=None,
        )

        from gpt_trader.core import Action as StandardAction

        assert decision.action == StandardAction.HOLD

    def test_decide_converts_buy(self):
        """Converts BUY action correctly."""
        config = HybridStrategyConfig()
        decisions = [
            HybridDecision(
                action=Action.BUY,
                symbol="BTC-USD",
                mode=TradingMode.SPOT_ONLY,
                quantity=Decimal("1"),
                reason="Test buy",
                confidence=0.8,
            )
        ]
        strategy = ConcreteHybridStrategy(config, decisions=decisions)

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[],
            equity=Decimal("10000"),
            product=None,
        )

        from gpt_trader.core import Action as StandardAction

        assert decision.action == StandardAction.BUY
        assert decision.reason == "Test buy"
        assert decision.confidence == 0.8

    def test_decide_converts_sell(self):
        """Converts SELL action correctly."""
        config = HybridStrategyConfig()
        decisions = [
            HybridDecision(
                action=Action.SELL,
                symbol="BTC-USD",
                mode=TradingMode.SPOT_ONLY,
            )
        ]
        strategy = ConcreteHybridStrategy(config, decisions=decisions)

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[],
            equity=Decimal("10000"),
            product=None,
        )

        from gpt_trader.core import Action as StandardAction

        assert decision.action == StandardAction.SELL

    def test_decide_skips_hold_decisions(self):
        """Skips HOLD decisions to find actionable one."""
        config = HybridStrategyConfig()
        decisions = [
            HybridDecision(
                action=Action.HOLD,
                symbol="BTC-USD",
                mode=TradingMode.SPOT_ONLY,
            ),
            HybridDecision(
                action=Action.BUY,
                symbol="BTC-USD",
                mode=TradingMode.SPOT_ONLY,
                reason="Second decision",
            ),
        ]
        strategy = ConcreteHybridStrategy(config, decisions=decisions)

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[],
            equity=Decimal("10000"),
            product=None,
        )

        from gpt_trader.core import Action as StandardAction

        assert decision.action == StandardAction.BUY
        assert decision.reason == "Second decision"

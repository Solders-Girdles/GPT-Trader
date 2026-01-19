"""Tests for HybridStrategyBase decision helper methods."""

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.hybrid.types import (
    Action,
    HybridStrategyConfig,
    TradingMode,
)
from tests.unit.gpt_trader.features.live_trade.strategies.hybrid.hybrid_strategy_test_helpers import (
    ConcreteHybridStrategy,
)


class TestHybridStrategyBaseHelpers:
    """Tests for helper methods."""

    def test_create_spot_decision(self):
        """Creates spot decision correctly."""
        config = HybridStrategyConfig()
        strategy = ConcreteHybridStrategy(config)

        decision = strategy.create_spot_decision(
            action=Action.BUY,
            quantity=Decimal("1.0"),
            reason="Test",
            confidence=0.7,
        )

        assert decision.action == Action.BUY
        assert decision.symbol == "BTC-USD"
        assert decision.mode == TradingMode.SPOT_ONLY
        assert decision.quantity == Decimal("1.0")
        assert decision.leverage == 1
        assert decision.reason == "Test"
        assert decision.confidence == 0.7

    def test_create_cfm_decision(self):
        """Creates CFM decision correctly."""
        config = HybridStrategyConfig(cfm_default_leverage=2)
        strategy = ConcreteHybridStrategy(config)

        decision = strategy.create_cfm_decision(
            action=Action.SELL,
            quantity=Decimal("0.5"),
            reason="Short signal",
        )

        assert decision.action == Action.SELL
        assert decision.symbol == "BTC-20DEC30-CDE"
        assert decision.mode == TradingMode.CFM_ONLY
        assert decision.quantity == Decimal("0.5")
        assert decision.leverage == 2  # Default from config

    def test_create_cfm_decision_with_custom_leverage(self):
        """Creates CFM decision with custom leverage."""
        config = HybridStrategyConfig(cfm_default_leverage=2)
        strategy = ConcreteHybridStrategy(config)

        decision = strategy.create_cfm_decision(
            action=Action.BUY,
            quantity=Decimal("0.5"),
            reason="Long with leverage",
            leverage=5,
        )

        assert decision.leverage == 5  # Override default

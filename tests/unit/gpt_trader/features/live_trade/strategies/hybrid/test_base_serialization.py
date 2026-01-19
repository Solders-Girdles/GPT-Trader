"""Tests for HybridStrategyBase serialization/deserialization."""

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.hybrid.types import HybridStrategyConfig
from tests.unit.gpt_trader.features.live_trade.strategies.hybrid.hybrid_strategy_test_helpers import (
    ConcreteHybridStrategy,
)


class TestHybridStrategyBaseSerialization:
    """Tests for state serialization."""

    def test_serialize_state(self):
        """Serializes state correctly."""
        config = HybridStrategyConfig(base_symbol="ETH")
        strategy = ConcreteHybridStrategy(config)
        strategy._position_state.spot_quantity = Decimal("1.5")
        strategy._position_state.spot_side = "long"

        state = strategy.serialize_state()

        assert "position_state" in state
        assert state["position_state"]["spot_quantity"] == "1.5"
        assert state["config"]["base_symbol"] == "ETH"

    def test_deserialize_state(self):
        """Deserializes state correctly."""
        config = HybridStrategyConfig()
        strategy = ConcreteHybridStrategy(config)

        state = {
            "position_state": {
                "spot_quantity": "2.0",
                "spot_entry_price": "49000",
                "spot_side": "long",
                "cfm_quantity": "1.0",
                "cfm_entry_price": "49500",
                "cfm_side": "short",
                "cfm_leverage": 3,
            }
        }
        strategy.deserialize_state(state)

        assert strategy.position_state.spot_quantity == Decimal("2.0")
        assert strategy.position_state.spot_entry_price == Decimal("49000")
        assert strategy.position_state.spot_side == "long"
        assert strategy.position_state.cfm_quantity == Decimal("1.0")
        assert strategy.position_state.cfm_side == "short"
        assert strategy.position_state.cfm_leverage == 3

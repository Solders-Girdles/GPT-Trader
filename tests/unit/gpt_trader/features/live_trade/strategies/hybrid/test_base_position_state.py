"""Tests for HybridStrategyBase position state updates."""

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.hybrid.types import HybridStrategyConfig
from tests.unit.gpt_trader.features.live_trade.strategies.hybrid.hybrid_strategy_test_helpers import (
    ConcreteHybridStrategy,
)


class TestHybridStrategyBasePositionState:
    """Tests for position state management."""

    def test_update_from_spot_position_dict(self):
        """Updates position state from spot position dict."""
        config = HybridStrategyConfig()
        strategy = ConcreteHybridStrategy(config)

        position_state = {
            "spot": {
                "quantity": "1.5",
                "entry_price": "50000",
                "side": "long",
            }
        }
        strategy._update_position_state_from_dict(position_state)

        assert strategy.position_state.spot_quantity == Decimal("1.5")
        assert strategy.position_state.spot_entry_price == Decimal("50000")
        assert strategy.position_state.spot_side == "long"

    def test_update_from_cfm_position_dict(self):
        """Updates position state from CFM position dict."""
        config = HybridStrategyConfig()
        strategy = ConcreteHybridStrategy(config)

        position_state = {
            "cfm": {
                "quantity": "0.5",
                "entry_price": "50500",
                "side": "short",
                "leverage": 3,
            }
        }
        strategy._update_position_state_from_dict(position_state)

        assert strategy.position_state.cfm_quantity == Decimal("0.5")
        assert strategy.position_state.cfm_entry_price == Decimal("50500")
        assert strategy.position_state.cfm_side == "short"
        assert strategy.position_state.cfm_leverage == 3

    def test_update_from_flat_dict_spot(self):
        """Updates from flat dict format with SPOT type."""
        config = HybridStrategyConfig()
        strategy = ConcreteHybridStrategy(config)

        position_state = {
            "quantity": "1.0",
            "entry_price": "50000",
            "product_type": "SPOT",
        }
        strategy._update_position_state_from_dict(position_state)

        assert strategy.position_state.spot_quantity == Decimal("1.0")
        assert strategy.position_state.spot_side == "long"

    def test_update_from_flat_dict_futures(self):
        """Updates from flat dict format with FUTURE type."""
        config = HybridStrategyConfig()
        strategy = ConcreteHybridStrategy(config)

        position_state = {
            "quantity": "-0.5",
            "entry_price": "50500",
            "product_type": "FUTURE",
            "leverage": 5,
        }
        strategy._update_position_state_from_dict(position_state)

        assert strategy.position_state.cfm_quantity == Decimal("0.5")
        assert strategy.position_state.cfm_side == "short"
        assert strategy.position_state.cfm_leverage == 5

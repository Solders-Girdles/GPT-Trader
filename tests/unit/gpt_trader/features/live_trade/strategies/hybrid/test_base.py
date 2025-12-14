"""Tests for HybridStrategyBase."""

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.strategies.hybrid.base import HybridStrategyBase
from gpt_trader.features.live_trade.strategies.hybrid.types import (
    Action,
    HybridDecision,
    HybridMarketData,
    HybridPositionState,
    HybridStrategyConfig,
    TradingMode,
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


class TestHybridStrategyBaseInit:
    """Tests for HybridStrategyBase initialization."""

    def test_init_default_config(self):
        """Initializes with default config."""
        config = HybridStrategyConfig()
        strategy = ConcreteHybridStrategy(config)
        assert strategy.config == config
        assert strategy.spot_symbol == "BTC-USD"
        assert strategy.cfm_symbol == "BTC-20DEC30-CDE"

    def test_init_custom_symbols(self):
        """Initializes with custom symbols."""
        config = HybridStrategyConfig(
            base_symbol="ETH",
            quote_currency="USDT",
        )
        strategy = ConcreteHybridStrategy(config)
        assert strategy.spot_symbol == "ETH-USDT"
        assert strategy.cfm_symbol == "ETH-20DEC30-CDE"

    def test_init_explicit_cfm_symbol(self):
        """Uses explicit CFM symbol if provided."""
        config = HybridStrategyConfig(
            cfm_symbol="CUSTOM-SYMBOL",
        )
        strategy = ConcreteHybridStrategy(config)
        assert strategy.cfm_symbol == "CUSTOM-SYMBOL"

    def test_init_position_state(self):
        """Initializes with empty position state."""
        config = HybridStrategyConfig()
        strategy = ConcreteHybridStrategy(config)
        assert strategy.position_state.spot_quantity == Decimal("0")
        assert strategy.position_state.cfm_quantity == Decimal("0")


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

        from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import Action as LegacyAction

        assert decision.action == LegacyAction.HOLD

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

        from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import Action as LegacyAction

        assert decision.action == LegacyAction.BUY
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

        from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import Action as LegacyAction

        assert decision.action == LegacyAction.SELL

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

        from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import Action as LegacyAction

        assert decision.action == LegacyAction.BUY
        assert decision.reason == "Second decision"


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


class TestHybridStrategyBasePositionSizing:
    """Tests for position sizing calculations."""

    def test_calculate_spot_position_size(self):
        """Calculates spot position size correctly."""
        config = HybridStrategyConfig(spot_position_size_pct=0.25)
        strategy = ConcreteHybridStrategy(config)

        size = strategy.calculate_position_size(
            equity=Decimal("10000"),
            mode=TradingMode.SPOT_ONLY,
        )

        assert size == Decimal("2500")  # 10000 * 0.25

    def test_calculate_cfm_position_size(self):
        """Calculates CFM position size correctly."""
        config = HybridStrategyConfig(cfm_position_size_pct=0.20)
        strategy = ConcreteHybridStrategy(config)

        size = strategy.calculate_position_size(
            equity=Decimal("10000"),
            mode=TradingMode.CFM_ONLY,
        )

        assert size == Decimal("2000")  # 10000 * 0.20

    def test_calculate_cfm_position_size_with_leverage(self):
        """Calculates CFM position size with leverage."""
        config = HybridStrategyConfig(cfm_position_size_pct=0.20)
        strategy = ConcreteHybridStrategy(config)

        size = strategy.calculate_position_size(
            equity=Decimal("10000"),
            mode=TradingMode.CFM_ONLY,
            leverage=5,
        )

        assert size == Decimal("10000")  # 10000 * 5 * 0.20

    def test_calculate_hybrid_position_size(self):
        """Calculates hybrid position size (average)."""
        config = HybridStrategyConfig(
            spot_position_size_pct=0.30,
            cfm_position_size_pct=0.20,
        )
        strategy = ConcreteHybridStrategy(config)

        size = strategy.calculate_position_size(
            equity=Decimal("10000"),
            mode=TradingMode.HYBRID,
        )

        assert size == Decimal("2500")  # 10000 * 0.25 (average of 0.30 and 0.20)


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

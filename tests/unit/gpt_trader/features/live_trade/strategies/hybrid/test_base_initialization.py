"""Tests for HybridStrategyBase initialization."""

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.hybrid.types import HybridStrategyConfig
from tests.unit.gpt_trader.features.live_trade.strategies.hybrid.hybrid_strategy_test_helpers import (
    ConcreteHybridStrategy,
)


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

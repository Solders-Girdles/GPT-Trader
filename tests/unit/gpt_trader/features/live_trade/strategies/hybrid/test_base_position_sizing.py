"""Tests for HybridStrategyBase position sizing calculations."""

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.hybrid.types import (
    HybridStrategyConfig,
    TradingMode,
)
from tests.unit.gpt_trader.features.live_trade.strategies.hybrid.hybrid_strategy_test_helpers import (
    ConcreteHybridStrategy,
)


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

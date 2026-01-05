"""Tests for OrderFlowSignal."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.features.live_trade.signals.order_flow import (
    OrderFlowSignal,
    OrderFlowSignalConfig,
)
from gpt_trader.features.live_trade.signals.protocol import StrategyContext
from gpt_trader.features.live_trade.signals.types import SignalType
from gpt_trader.features.live_trade.strategies.base import MarketDataContext


class TestOrderFlowSignalConfig:
    """Tests for OrderFlowSignalConfig."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = OrderFlowSignalConfig()
        assert config.aggressor_threshold_bullish == 0.6
        assert config.aggressor_threshold_bearish == 0.4
        assert config.min_trades == 10
        assert config.volume_weight is True

    def test_custom_values(self) -> None:
        """Test custom config values."""
        config = OrderFlowSignalConfig(
            aggressor_threshold_bullish=0.7,
            aggressor_threshold_bearish=0.3,
            min_trades=20,
        )
        assert config.aggressor_threshold_bullish == 0.7
        assert config.aggressor_threshold_bearish == 0.3
        assert config.min_trades == 20


class TestOrderFlowSignal:
    """Tests for OrderFlowSignal."""

    @pytest.fixture
    def signal(self) -> OrderFlowSignal:
        """Create a signal with default config."""
        return OrderFlowSignal()

    @pytest.fixture
    def base_context(self) -> StrategyContext:
        """Create a basic strategy context."""
        return StrategyContext(
            symbol="BTC-USD",
            current_mark=Decimal("50000"),
            position_state=None,
            recent_marks=[Decimal("49900"), Decimal("49950"), Decimal("50000")],
            equity=Decimal("10000"),
            product=None,
            market_data=None,
        )

    def test_no_market_data_returns_neutral(
        self, signal: OrderFlowSignal, base_context: StrategyContext
    ) -> None:
        """Test that missing market_data returns neutral signal."""
        result = signal.generate(base_context)

        assert result.name == "order_flow"
        assert result.type == SignalType.ORDER_FLOW
        assert result.strength == 0.0
        assert result.confidence == 0.0
        assert result.metadata["reason"] == "no_market_data"

    def test_no_trade_stats_returns_neutral(
        self, signal: OrderFlowSignal, base_context: StrategyContext
    ) -> None:
        """Test that missing trade_stats returns neutral signal."""
        base_context.market_data = MarketDataContext(trade_volume_stats=None)
        result = signal.generate(base_context)

        assert result.strength == 0.0
        assert result.confidence == 0.0
        assert result.metadata["reason"] == "no_trade_stats"

    def test_insufficient_trades_returns_neutral(
        self, signal: OrderFlowSignal, base_context: StrategyContext
    ) -> None:
        """Test that too few trades returns neutral signal."""
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 5, "aggressor_ratio": 0.7}
        )
        result = signal.generate(base_context)

        assert result.strength == 0.0
        assert result.metadata["reason"] == "insufficient_trades"
        assert result.metadata["trade_count"] == 5

    def test_high_buy_aggression_is_bullish(
        self, signal: OrderFlowSignal, base_context: StrategyContext
    ) -> None:
        """Test that high buy aggression generates bullish signal."""
        base_context.market_data = MarketDataContext(
            trade_volume_stats={
                "count": 50,
                "aggressor_ratio": 0.75,
                "volume": Decimal("100"),
                "vwap": Decimal("50000"),
            }
        )
        result = signal.generate(base_context)

        assert result.strength > 0  # Bullish
        assert result.confidence > 0.3
        assert result.metadata["reason"] == "buy_aggression"
        assert result.metadata["aggressor_ratio"] == 0.75

    def test_high_sell_aggression_is_bearish(
        self, signal: OrderFlowSignal, base_context: StrategyContext
    ) -> None:
        """Test that high sell aggression generates bearish signal."""
        base_context.market_data = MarketDataContext(
            trade_volume_stats={
                "count": 50,
                "aggressor_ratio": 0.25,
                "volume": Decimal("100"),
                "vwap": Decimal("50000"),
            }
        )
        result = signal.generate(base_context)

        assert result.strength < 0  # Bearish
        assert result.confidence > 0.3
        assert result.metadata["reason"] == "sell_aggression"

    def test_balanced_flow_is_neutral(
        self, signal: OrderFlowSignal, base_context: StrategyContext
    ) -> None:
        """Test that balanced flow generates neutral signal."""
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 50, "aggressor_ratio": 0.5}
        )
        result = signal.generate(base_context)

        assert result.strength == 0.0
        assert result.metadata["reason"] == "balanced_flow"

    def test_confidence_increases_with_trade_count(
        self, signal: OrderFlowSignal, base_context: StrategyContext
    ) -> None:
        """Test that more trades increase confidence."""
        # Low trade count
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 15, "aggressor_ratio": 0.7}
        )
        low_result = signal.generate(base_context)

        # High trade count
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 100, "aggressor_ratio": 0.7}
        )
        high_result = signal.generate(base_context)

        assert high_result.confidence > low_result.confidence

    def test_strength_scales_with_aggressor_ratio(
        self, signal: OrderFlowSignal, base_context: StrategyContext
    ) -> None:
        """Test that strength scales with aggressor ratio magnitude."""
        # Moderate aggression
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 50, "aggressor_ratio": 0.65}
        )
        moderate_result = signal.generate(base_context)

        # Strong aggression
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 50, "aggressor_ratio": 0.90}
        )
        strong_result = signal.generate(base_context)

        assert strong_result.strength > moderate_result.strength

    def test_custom_thresholds(self, base_context: StrategyContext) -> None:
        """Test signal with custom thresholds."""
        custom_signal = OrderFlowSignal(
            OrderFlowSignalConfig(
                aggressor_threshold_bullish=0.7,
                aggressor_threshold_bearish=0.3,
            )
        )

        # 0.65 would be bullish with default 0.6, but neutral with 0.7
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 50, "aggressor_ratio": 0.65}
        )
        result = custom_signal.generate(base_context)

        assert result.strength == 0.0
        assert result.metadata["reason"] == "balanced_flow"

"""Tests for OrderbookImbalanceSignal."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.features.live_trade.signals.orderbook_imbalance import (
    OrderbookImbalanceSignal,
    OrderbookImbalanceSignalConfig,
)
from gpt_trader.features.live_trade.signals.protocol import StrategyContext
from gpt_trader.features.live_trade.signals.types import SignalType
from gpt_trader.features.live_trade.strategies.base import MarketDataContext


class MockDepthSnapshot:
    """Mock orderbook depth snapshot."""

    def __init__(
        self,
        bid_depth: Decimal,
        ask_depth: Decimal,
        spread_bps: float | None = None,
    ) -> None:
        self._bid_depth = bid_depth
        self._ask_depth = ask_depth
        self.spread_bps = spread_bps

    def get_depth(self, levels: int) -> tuple[Decimal, Decimal]:
        """Return bid and ask depth."""
        return self._bid_depth, self._ask_depth


class TestOrderbookImbalanceSignalConfig:
    """Tests for OrderbookImbalanceSignalConfig."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = OrderbookImbalanceSignalConfig()
        assert config.levels == 5
        assert config.imbalance_threshold == 0.2
        assert config.strong_imbalance_threshold == 0.5

    def test_custom_values(self) -> None:
        """Test custom config values."""
        config = OrderbookImbalanceSignalConfig(
            levels=10,
            imbalance_threshold=0.3,
            strong_imbalance_threshold=0.6,
        )
        assert config.levels == 10
        assert config.imbalance_threshold == 0.3


class TestOrderbookImbalanceSignal:
    """Tests for OrderbookImbalanceSignal."""

    @pytest.fixture
    def signal(self) -> OrderbookImbalanceSignal:
        """Create a signal with default config."""
        return OrderbookImbalanceSignal()

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
        self, signal: OrderbookImbalanceSignal, base_context: StrategyContext
    ) -> None:
        """Test that missing market_data returns neutral signal."""
        result = signal.generate(base_context)

        assert result.name == "orderbook_imbalance"
        assert result.type == SignalType.MICROSTRUCTURE
        assert result.strength == 0.0
        assert result.confidence == 0.0
        assert result.metadata["reason"] == "no_market_data"

    def test_no_orderbook_returns_neutral(
        self, signal: OrderbookImbalanceSignal, base_context: StrategyContext
    ) -> None:
        """Test that missing orderbook returns neutral signal."""
        base_context.market_data = MarketDataContext(orderbook_snapshot=None)
        result = signal.generate(base_context)

        assert result.strength == 0.0
        assert result.metadata["reason"] == "no_orderbook"

    def test_bid_heavy_is_bullish(
        self, signal: OrderbookImbalanceSignal, base_context: StrategyContext
    ) -> None:
        """Test that bid-heavy orderbook generates bullish signal."""
        snapshot = MockDepthSnapshot(
            bid_depth=Decimal("100"),
            ask_depth=Decimal("50"),
            spread_bps=5.0,
        )
        base_context.market_data = MarketDataContext(orderbook_snapshot=snapshot)

        result = signal.generate(base_context)

        # Imbalance = (100-50)/(100+50) = 0.33
        assert result.strength > 0  # Bullish
        assert result.metadata["reason"] == "bid_heavy"
        assert result.metadata["imbalance"] > 0

    def test_ask_heavy_is_bearish(
        self, signal: OrderbookImbalanceSignal, base_context: StrategyContext
    ) -> None:
        """Test that ask-heavy orderbook generates bearish signal."""
        snapshot = MockDepthSnapshot(
            bid_depth=Decimal("50"),
            ask_depth=Decimal("100"),
            spread_bps=5.0,
        )
        base_context.market_data = MarketDataContext(orderbook_snapshot=snapshot)

        result = signal.generate(base_context)

        # Imbalance = (50-100)/(50+100) = -0.33
        assert result.strength < 0  # Bearish
        assert result.metadata["reason"] == "ask_heavy"
        assert result.metadata["imbalance"] < 0

    def test_balanced_is_neutral(
        self, signal: OrderbookImbalanceSignal, base_context: StrategyContext
    ) -> None:
        """Test that balanced orderbook generates neutral signal."""
        snapshot = MockDepthSnapshot(
            bid_depth=Decimal("100"),
            ask_depth=Decimal("100"),
            spread_bps=5.0,
        )
        base_context.market_data = MarketDataContext(orderbook_snapshot=snapshot)

        result = signal.generate(base_context)

        # Imbalance = 0
        assert result.strength == 0.0
        assert result.metadata["reason"] == "balanced"

    def test_wide_spread_reduces_confidence(
        self, signal: OrderbookImbalanceSignal, base_context: StrategyContext
    ) -> None:
        """Test that wide spread reduces confidence."""
        # Tight spread
        tight_snapshot = MockDepthSnapshot(
            bid_depth=Decimal("100"),
            ask_depth=Decimal("50"),
            spread_bps=5.0,
        )
        base_context.market_data = MarketDataContext(orderbook_snapshot=tight_snapshot)
        tight_result = signal.generate(base_context)

        # Wide spread
        wide_snapshot = MockDepthSnapshot(
            bid_depth=Decimal("100"),
            ask_depth=Decimal("50"),
            spread_bps=25.0,
        )
        base_context.market_data = MarketDataContext(orderbook_snapshot=wide_snapshot)
        wide_result = signal.generate(base_context)

        assert wide_result.confidence < tight_result.confidence

    def test_strong_imbalance_increases_confidence(
        self, signal: OrderbookImbalanceSignal, base_context: StrategyContext
    ) -> None:
        """Test that stronger imbalance increases confidence."""
        # Moderate imbalance
        moderate_snapshot = MockDepthSnapshot(
            bid_depth=Decimal("60"),
            ask_depth=Decimal("40"),
            spread_bps=5.0,
        )
        base_context.market_data = MarketDataContext(orderbook_snapshot=moderate_snapshot)
        moderate_result = signal.generate(base_context)

        # Strong imbalance
        strong_snapshot = MockDepthSnapshot(
            bid_depth=Decimal("80"),
            ask_depth=Decimal("20"),
            spread_bps=5.0,
        )
        base_context.market_data = MarketDataContext(orderbook_snapshot=strong_snapshot)
        strong_result = signal.generate(base_context)

        assert strong_result.confidence > moderate_result.confidence

    def test_zero_depth_returns_neutral(
        self, signal: OrderbookImbalanceSignal, base_context: StrategyContext
    ) -> None:
        """Test that zero total depth returns neutral signal."""
        snapshot = MockDepthSnapshot(
            bid_depth=Decimal("0"),
            ask_depth=Decimal("0"),
            spread_bps=5.0,
        )
        base_context.market_data = MarketDataContext(orderbook_snapshot=snapshot)

        result = signal.generate(base_context)

        assert result.strength == 0.0
        assert result.metadata["reason"] == "zero_depth"

    def test_custom_thresholds(self, base_context: StrategyContext) -> None:
        """Test signal with custom thresholds."""
        custom_signal = OrderbookImbalanceSignal(
            OrderbookImbalanceSignalConfig(
                imbalance_threshold=0.4,
                strong_imbalance_threshold=0.7,
            )
        )

        # Imbalance of 0.33 is above default 0.2 but below custom 0.4
        snapshot = MockDepthSnapshot(
            bid_depth=Decimal("100"),
            ask_depth=Decimal("50"),
            spread_bps=5.0,
        )
        base_context.market_data = MarketDataContext(orderbook_snapshot=snapshot)
        result = custom_signal.generate(base_context)

        # Should be neutral with higher threshold
        assert result.strength == 0.0
        assert result.metadata["reason"] == "balanced"

    def test_levels_config_passed_to_depth(self, base_context: StrategyContext) -> None:
        """Test that levels config is passed to get_depth."""
        custom_signal = OrderbookImbalanceSignal(
            OrderbookImbalanceSignalConfig(levels=10)
        )

        snapshot = MockDepthSnapshot(
            bid_depth=Decimal("100"),
            ask_depth=Decimal("50"),
            spread_bps=5.0,
        )
        base_context.market_data = MarketDataContext(orderbook_snapshot=snapshot)
        result = custom_signal.generate(base_context)

        assert result.metadata["levels_analyzed"] == 10

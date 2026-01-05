"""Tests for VWAPSignal."""

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.signals.protocol import StrategyContext
from gpt_trader.features.live_trade.signals.types import SignalType
from gpt_trader.features.live_trade.signals.vwap import (
    VWAPSignal,
    VWAPSignalConfig,
)
from gpt_trader.features.live_trade.strategies.base import MarketDataContext


class TestVWAPSignalConfig:
    """Tests for VWAPSignalConfig."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = VWAPSignalConfig()
        assert config.deviation_threshold == 0.01
        assert config.strong_deviation_threshold == 0.025
        assert config.min_trades == 20

    def test_custom_values(self) -> None:
        """Test custom config values."""
        config = VWAPSignalConfig(
            deviation_threshold=0.005,
            strong_deviation_threshold=0.02,
            min_trades=50,
        )
        assert config.deviation_threshold == 0.005
        assert config.min_trades == 50


class TestVWAPSignal:
    """Tests for VWAPSignal."""

    @pytest.fixture
    def signal(self) -> VWAPSignal:
        """Create a signal with default config."""
        return VWAPSignal()

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
        self, signal: VWAPSignal, base_context: StrategyContext
    ) -> None:
        """Test that missing market_data returns neutral signal."""
        result = signal.generate(base_context)

        assert result.name == "vwap_deviation"
        assert result.type == SignalType.MEAN_REVERSION
        assert result.strength == 0.0
        assert result.confidence == 0.0
        assert result.metadata["reason"] == "no_market_data"

    def test_no_trade_stats_returns_neutral(
        self, signal: VWAPSignal, base_context: StrategyContext
    ) -> None:
        """Test that missing trade_stats returns neutral signal."""
        base_context.market_data = MarketDataContext(trade_volume_stats=None)
        result = signal.generate(base_context)

        assert result.strength == 0.0
        assert result.metadata["reason"] == "no_trade_stats"

    def test_no_vwap_returns_neutral(
        self, signal: VWAPSignal, base_context: StrategyContext
    ) -> None:
        """Test that missing vwap returns neutral signal."""
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 50}
        )
        result = signal.generate(base_context)

        assert result.strength == 0.0
        assert result.metadata["reason"] == "no_vwap"

    def test_insufficient_trades_returns_neutral(
        self, signal: VWAPSignal, base_context: StrategyContext
    ) -> None:
        """Test that too few trades returns neutral signal."""
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 10, "vwap": Decimal("50000")}
        )
        result = signal.generate(base_context)

        assert result.strength == 0.0
        assert result.metadata["reason"] == "insufficient_trades"
        assert result.metadata["trade_count"] == 10

    def test_price_below_vwap_is_bullish(
        self, signal: VWAPSignal, base_context: StrategyContext
    ) -> None:
        """Test that price below VWAP generates bullish signal."""
        # Price is 50000, VWAP is 51000 (price is ~2% below)
        base_context.market_data = MarketDataContext(
            trade_volume_stats={
                "count": 50,
                "vwap": Decimal("51000"),
                "volume": Decimal("1000"),
            }
        )
        result = signal.generate(base_context)

        assert result.strength > 0  # Bullish (mean reversion buy)
        assert result.confidence > 0.3
        assert result.metadata["reason"] == "below_vwap"
        assert result.metadata["deviation_pct"] < 0

    def test_price_above_vwap_is_bearish(
        self, signal: VWAPSignal, base_context: StrategyContext
    ) -> None:
        """Test that price above VWAP generates bearish signal."""
        # Price is 50000, VWAP is 49000 (price is ~2% above)
        base_context.market_data = MarketDataContext(
            trade_volume_stats={
                "count": 50,
                "vwap": Decimal("49000"),
                "volume": Decimal("1000"),
            }
        )
        result = signal.generate(base_context)

        assert result.strength < 0  # Bearish (mean reversion sell)
        assert result.confidence > 0.3
        assert result.metadata["reason"] == "above_vwap"
        assert result.metadata["deviation_pct"] > 0

    def test_price_near_vwap_is_neutral(
        self, signal: VWAPSignal, base_context: StrategyContext
    ) -> None:
        """Test that price near VWAP generates neutral signal."""
        # Price is 50000, VWAP is 50100 (only 0.2% away)
        base_context.market_data = MarketDataContext(
            trade_volume_stats={
                "count": 50,
                "vwap": Decimal("50100"),
            }
        )
        result = signal.generate(base_context)

        assert result.strength == 0.0
        assert result.metadata["reason"] == "near_vwap"

    def test_confidence_increases_with_trade_count(
        self, signal: VWAPSignal, base_context: StrategyContext
    ) -> None:
        """Test that more trades increase confidence."""
        # Low trade count
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 25, "vwap": Decimal("51000")}
        )
        low_result = signal.generate(base_context)

        # High trade count
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 150, "vwap": Decimal("51000")}
        )
        high_result = signal.generate(base_context)

        assert high_result.confidence > low_result.confidence

    def test_strength_scales_with_deviation(
        self, signal: VWAPSignal, base_context: StrategyContext
    ) -> None:
        """Test that strength scales with deviation magnitude."""
        # Small deviation
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 50, "vwap": Decimal("50600")}  # ~1.2% above
        )
        small_result = signal.generate(base_context)

        # Large deviation
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 50, "vwap": Decimal("52500")}  # ~5% above
        )
        large_result = signal.generate(base_context)

        assert abs(large_result.strength) > abs(small_result.strength)

    def test_strong_deviation_increases_confidence(
        self, signal: VWAPSignal, base_context: StrategyContext
    ) -> None:
        """Test that stronger deviations have higher confidence."""
        # Moderate deviation (1.5%)
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 50, "vwap": Decimal("50750")}
        )
        moderate_result = signal.generate(base_context)

        # Strong deviation (3%)
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 50, "vwap": Decimal("51500")}
        )
        strong_result = signal.generate(base_context)

        assert strong_result.confidence > moderate_result.confidence

    def test_zero_vwap_returns_neutral(
        self, signal: VWAPSignal, base_context: StrategyContext
    ) -> None:
        """Test that zero vwap returns neutral signal."""
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 50, "vwap": Decimal("0")}
        )
        result = signal.generate(base_context)

        assert result.strength == 0.0
        assert result.metadata["reason"] == "no_vwap"

    def test_custom_thresholds(self, base_context: StrategyContext) -> None:
        """Test signal with custom thresholds."""
        custom_signal = VWAPSignal(
            VWAPSignalConfig(
                deviation_threshold=0.02,  # 2%
                min_trades=10,
            )
        )

        # 1.5% deviation would trigger default but not custom
        base_context.market_data = MarketDataContext(
            trade_volume_stats={"count": 50, "vwap": Decimal("50750")}
        )
        result = custom_signal.generate(base_context)

        assert result.strength == 0.0
        assert result.metadata["reason"] == "near_vwap"

    def test_metadata_contains_expected_fields(
        self, signal: VWAPSignal, base_context: StrategyContext
    ) -> None:
        """Test that metadata contains all expected fields."""
        base_context.market_data = MarketDataContext(
            trade_volume_stats={
                "count": 50,
                "vwap": Decimal("51000"),
                "volume": Decimal("1000"),
                "avg_size": Decimal("20"),
            }
        )
        result = signal.generate(base_context)

        assert "vwap" in result.metadata
        assert "current_price" in result.metadata
        assert "deviation_pct" in result.metadata
        assert "trade_count" in result.metadata
        assert "volume" in result.metadata
        assert "avg_size" in result.metadata
        assert "reason" in result.metadata

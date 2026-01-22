"""Tests for IndicatorBundle, Z-Score, and performance benchmarks."""

from decimal import Decimal

from gpt_trader.features.live_trade.indicators import simple_moving_average
from gpt_trader.features.live_trade.stateful_indicators import (
    IndicatorBundle,
    OnlineEMA,
    OnlineSMA,
    OnlineZScore,
)


class TestIndicatorBundle:
    """Test unified indicator bundle."""

    def test_create_with_defaults(self) -> None:
        """Test bundle creation with default parameters."""
        bundle = IndicatorBundle.create("BTC-USD")

        assert bundle.symbol == "BTC-USD"
        assert bundle.short_sma.period == 5
        assert bundle.long_sma.period == 20
        assert bundle.rsi.period == 14

    def test_create_with_custom_params(self) -> None:
        """Test bundle creation with custom parameters."""
        bundle = IndicatorBundle.create(
            "ETH-USD",
            short_period=3,
            long_period=10,
            rsi_period=7,
        )

        assert bundle.short_sma.period == 3
        assert bundle.long_sma.period == 10
        assert bundle.rsi.period == 7

    def test_update_returns_all_values(self) -> None:
        """Test update returns dict of all indicator values."""
        bundle = IndicatorBundle.create("BTC-USD", short_period=2, long_period=3)

        # First few updates during warmup
        result = bundle.update(Decimal("100"))
        assert "short_sma" in result
        assert "long_sma" in result
        assert "rsi" in result
        assert result["price"] == Decimal("100")

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Test serialization preserves all indicator states."""
        bundle = IndicatorBundle.create("BTC-USD", short_period=3, long_period=5)

        # Add some data
        for price in [Decimal(str(p)) for p in [100, 102, 101, 103, 105, 104]]:
            bundle.update(price)

        serialized = bundle.serialize()
        restored = IndicatorBundle.deserialize(serialized)

        # Verify state matches
        assert restored.symbol == bundle.symbol
        assert restored.short_sma.value == bundle.short_sma.value
        assert restored.long_sma.value == bundle.long_sma.value

        # Verify continued updates match
        original_result = bundle.update(Decimal("106"))
        restored_result = restored.update(Decimal("106"))

        assert original_result["short_sma"] == restored_result["short_sma"]
        assert original_result["long_sma"] == restored_result["long_sma"]


class TestOnlineZScore:
    """Test online Z-Score calculation."""

    def test_insufficient_data_returns_none(self) -> None:
        """Test Z-Score returns None with insufficient data."""
        zscore = OnlineZScore()
        assert zscore.update(Decimal("100")) is None  # Need at least 2 for std_dev

    def test_z_score_calculation(self) -> None:
        """Test Z-Score is calculated correctly."""
        zscore = OnlineZScore()
        # Add values to build up statistics
        for v in [Decimal("3"), Decimal("5"), Decimal("7")]:
            zscore.update(v)

        # Add value above the mean - should have positive Z-Score
        result = zscore.update(Decimal("9"))
        assert result is not None
        # 9 is above the mean, so Z-Score should be positive
        assert result > Decimal("0")

    def test_windowed_z_score(self) -> None:
        """Test Z-Score with rolling window."""
        zscore = OnlineZScore(lookback=3)

        # Fill window with identical values first
        zscore.update(Decimal("20"))
        zscore.update(Decimal("20"))
        zscore.update(Decimal("20"))

        # Window = [20, 20, 20], mean = 20, std_dev = 0
        # Z-Score of value at mean with zero std_dev should return 0
        result = zscore.update(Decimal("20"))
        assert result is not None
        assert result == Decimal("0")  # At the mean with zero variance

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Test serialization preserves Z-Score state."""
        zscore = OnlineZScore()
        for v in [Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40")]:
            zscore.update(v)

        serialized = zscore.serialize()
        restored = OnlineZScore.deserialize(serialized)

        assert restored.welford.count == zscore.welford.count
        assert restored.welford.mean == zscore.welford.mean


class TestOnlineSMA:
    """Test online Simple Moving Average."""

    def test_warmup_period(self) -> None:
        """Test SMA during warmup before window is full."""
        sma = OnlineSMA(period=3)
        assert sma.update(Decimal("10")) == Decimal("10")  # Single value mean
        assert sma.update(Decimal("20")) == Decimal("15")  # Two values mean
        assert not sma.is_ready

    def test_full_window(self) -> None:
        """Test SMA when window is full."""
        sma = OnlineSMA(period=3)
        sma.update(Decimal("10"))
        sma.update(Decimal("20"))
        result = sma.update(Decimal("30"))

        assert sma.is_ready
        assert result == Decimal("20")  # (10 + 20 + 30) / 3

    def test_matches_stateless_sma(self) -> None:
        """Test that online SMA matches stateless implementation."""
        prices = [Decimal(str(p)) for p in [100, 102, 101, 103, 105, 104, 106, 108]]
        period = 5

        # Online calculation
        online_sma = OnlineSMA(period=period)
        online_results = []
        for price in prices:
            online_results.append(online_sma.update(price))

        # Stateless calculation
        for i in range(period, len(prices) + 1):
            stateless_result = simple_moving_average(prices[:i], period)
            # Compare at corresponding indices
            online_result = online_results[i - 1]
            if online_result is not None and stateless_result is not None:
                assert abs(online_result - stateless_result) < Decimal("0.0001")

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Test serialization preserves SMA state."""
        sma = OnlineSMA(period=5)
        for i in range(1, 6):
            sma.update(Decimal(str(i * 10)))

        serialized = sma.serialize()
        restored = OnlineSMA.deserialize(serialized)

        # Verify next update produces same result
        original_next = sma.update(Decimal("60"))
        restored_next = restored.update(Decimal("60"))
        assert original_next == restored_next


class TestOnlineEMA:
    """Test online Exponential Moving Average."""

    def test_warmup_period(self) -> None:
        """Test EMA during warmup before initialization."""
        ema = OnlineEMA(period=3)
        assert ema.update(Decimal("10")) is None
        assert ema.update(Decimal("20")) is None
        assert not ema.initialized

    def test_initialization(self) -> None:
        """Test EMA initializes with SMA of warmup values."""
        ema = OnlineEMA(period=3)
        ema.update(Decimal("10"))
        ema.update(Decimal("20"))
        result = ema.update(Decimal("30"))

        assert ema.initialized
        # Initial EMA = SMA of first 3 values = 20
        assert result == Decimal("20")

    def test_subsequent_updates(self) -> None:
        """Test EMA updates after initialization."""
        ema = OnlineEMA(period=3)
        for price in [Decimal("10"), Decimal("20"), Decimal("30")]:
            ema.update(price)

        # Update with new price
        result = ema.update(Decimal("40"))

        # EMA formula: k = 2/(3+1) = 0.5
        # new_ema = 40 * 0.5 + 20 * 0.5 = 30
        assert result == Decimal("30")

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Test serialization preserves EMA state."""
        ema = OnlineEMA(period=5)
        for i in range(1, 8):
            ema.update(Decimal(str(i * 10)))

        serialized = ema.serialize()
        restored = OnlineEMA.deserialize(serialized)

        # Verify continued updates produce same results
        original_next = ema.update(Decimal("80"))
        restored_next = restored.update(Decimal("80"))
        assert original_next == restored_next

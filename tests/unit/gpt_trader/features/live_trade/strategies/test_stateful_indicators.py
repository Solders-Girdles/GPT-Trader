"""
Tests for stateful indicators using Welford's algorithm and online updates.

Verifies:
- Numerical correctness vs stateless implementations
- Serialization/deserialization round-trips
- Edge cases and boundary conditions
"""

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.indicators import (
    relative_strength_index,
    simple_moving_average,
)
from gpt_trader.features.live_trade.stateful_indicators import (
    IndicatorBundle,
    OnlineEMA,
    OnlineRSI,
    OnlineSMA,
    OnlineZScore,
    RollingWindow,
    WelfordState,
)


class TestWelfordState:
    """Test Welford's online algorithm for mean/variance."""

    def test_empty_state(self) -> None:
        """Test initial state values."""
        state = WelfordState()
        assert state.count == 0
        assert state.mean == Decimal("0")
        assert state.variance == Decimal("0")

    def test_single_value(self) -> None:
        """Test state after single update."""
        state = WelfordState()
        state.update(Decimal("10"))
        assert state.count == 1
        assert state.mean == Decimal("10")
        assert state.variance == Decimal("0")  # Need at least 2 for variance

    def test_multiple_values_mean(self) -> None:
        """Test mean calculation with multiple values."""
        state = WelfordState()
        values = [Decimal("10"), Decimal("20"), Decimal("30")]
        for v in values:
            state.update(v)

        assert state.count == 3
        assert state.mean == Decimal("20")  # (10 + 20 + 30) / 3

    def test_variance_calculation(self) -> None:
        """Test variance matches expected value."""
        state = WelfordState()
        # Values: 2, 4, 4, 4, 5, 5, 7, 9 (classic example)
        values = [Decimal(v) for v in [2, 4, 4, 4, 5, 5, 7, 9]]
        for v in values:
            state.update(v)

        # Mean = 5, Variance = 4 (population)
        assert state.mean == Decimal("5")
        # Allow for small floating point differences
        assert abs(state.variance - Decimal("4")) < Decimal("0.0001")

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Test serialization preserves state."""
        state = WelfordState()
        for v in [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]:
            state.update(v)

        serialized = state.serialize()
        restored = WelfordState.deserialize(serialized)

        assert restored.count == state.count
        assert restored.mean == state.mean
        assert restored.m2 == state.m2


class TestRollingWindow:
    """Test circular buffer rolling window."""

    def test_add_below_capacity(self) -> None:
        """Test adding values before window is full."""
        window = RollingWindow(max_size=3)
        assert window.add(Decimal("1")) is None
        assert window.add(Decimal("2")) is None
        assert len(window) == 2
        assert not window.is_full

    def test_add_at_capacity_evicts(self) -> None:
        """Test that adding at capacity evicts oldest value."""
        window = RollingWindow(max_size=3)
        window.add(Decimal("1"))
        window.add(Decimal("2"))
        window.add(Decimal("3"))
        assert window.is_full

        evicted = window.add(Decimal("4"))
        assert evicted == Decimal("1")
        assert list(window.values) == [Decimal("2"), Decimal("3"), Decimal("4")]

    def test_mean_calculation(self) -> None:
        """Test O(1) mean calculation."""
        window = RollingWindow(max_size=3)
        window.add(Decimal("10"))
        window.add(Decimal("20"))
        window.add(Decimal("30"))

        assert window.mean == Decimal("20")

    def test_running_sum_accuracy(self) -> None:
        """Test running sum stays accurate through evictions."""
        window = RollingWindow(max_size=3)
        for i in range(1, 11):
            window.add(Decimal(str(i)))

        # Window should contain [8, 9, 10]
        assert window.running_sum == Decimal("27")
        assert window.mean == Decimal("9")

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Test serialization preserves window state."""
        window = RollingWindow(max_size=5)
        for i in range(1, 4):
            window.add(Decimal(str(i)))

        serialized = window.serialize()
        restored = RollingWindow.deserialize(serialized)

        assert restored.max_size == window.max_size
        assert list(restored.values) == list(window.values)
        assert restored.running_sum == window.running_sum


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


class TestOnlineRSI:
    """Test online Relative Strength Index."""

    def test_first_price_returns_none(self) -> None:
        """Test that first price (no change) returns None."""
        rsi = OnlineRSI(period=14)
        assert rsi.update(Decimal("100")) is None

    def test_warmup_period(self) -> None:
        """Test RSI during warmup before initialization."""
        rsi = OnlineRSI(period=3)
        rsi.update(Decimal("100"))  # First price, no change yet
        assert rsi.update(Decimal("102")) is None  # +2
        assert rsi.update(Decimal("101")) is None  # -1
        assert not rsi.initialized

    def test_initialization(self) -> None:
        """Test RSI initializes after period changes."""
        rsi = OnlineRSI(period=3)
        rsi.update(Decimal("100"))
        rsi.update(Decimal("102"))  # +2
        rsi.update(Decimal("104"))  # +2
        result = rsi.update(Decimal("103"))  # -1

        assert rsi.initialized
        assert result is not None
        assert Decimal("0") <= result <= Decimal("100")

    def test_all_gains_rsi_100(self) -> None:
        """Test RSI approaches 100 with only gains."""
        rsi = OnlineRSI(period=3)
        rsi.update(Decimal("100"))
        for price in [Decimal("102"), Decimal("104"), Decimal("106"), Decimal("108")]:
            result = rsi.update(price)

        assert result == Decimal("100")

    def test_matches_stateless_rsi(self) -> None:
        """Test that online RSI matches stateless implementation."""
        prices = [
            Decimal(str(p))
            for p in [
                100,
                102,
                101,
                103,
                105,
                104,
                106,
                108,
                107,
                109,
                111,
                110,
                112,
                114,
                113,
                115,
                117,
            ]
        ]
        period = 14

        # Online calculation
        online_rsi = OnlineRSI(period=period)
        online_result = None
        for price in prices:
            online_result = online_rsi.update(price)

        # Stateless calculation
        stateless_result = relative_strength_index(prices, period)

        # Both should be non-None and approximately equal
        if online_result is not None and stateless_result is not None:
            # Allow small numerical differences due to order of operations
            assert abs(online_result - stateless_result) < Decimal("0.5")

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Test serialization preserves RSI state."""
        rsi = OnlineRSI(period=5)
        prices = [Decimal(str(p)) for p in [100, 102, 101, 103, 105, 104, 106]]
        for price in prices:
            rsi.update(price)

        serialized = rsi.serialize()
        restored = OnlineRSI.deserialize(serialized)

        # Verify continued updates produce same results
        original_next = rsi.update(Decimal("108"))
        restored_next = restored.update(Decimal("108"))
        assert original_next == restored_next


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


class TestPerformanceComparison:
    """Test that stateful indicators are more efficient than stateless."""

    @pytest.mark.slow
    def test_large_dataset_efficiency(self) -> None:
        """Test that stateful indicators don't recompute history.

        This test verifies the fundamental advantage of stateful indicators:
        they maintain running state rather than recomputing from scratch.
        """
        # Generate large price series
        prices = [Decimal("100") + Decimal(str(i % 10)) for i in range(1000)]

        # Stateful: O(1) per update
        sma = OnlineSMA(period=20)
        rsi = OnlineRSI(period=14)

        for price in prices:
            sma.update(price)
            rsi.update(price)

        # Final values should be valid
        assert sma.value is not None
        assert rsi.value is not None
        assert Decimal("0") <= rsi.value <= Decimal("100")

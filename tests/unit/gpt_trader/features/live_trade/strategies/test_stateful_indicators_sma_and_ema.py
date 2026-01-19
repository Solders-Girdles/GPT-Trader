"""Tests for OnlineSMA and OnlineEMA stateful indicators."""

from decimal import Decimal

from gpt_trader.features.live_trade.indicators import simple_moving_average
from gpt_trader.features.live_trade.stateful_indicators import OnlineEMA, OnlineSMA


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

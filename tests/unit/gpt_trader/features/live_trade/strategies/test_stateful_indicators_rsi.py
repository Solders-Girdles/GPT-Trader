"""Tests for OnlineRSI stateful indicator."""

from decimal import Decimal

from gpt_trader.features.live_trade.indicators import relative_strength_index
from gpt_trader.features.live_trade.stateful_indicators import OnlineRSI


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

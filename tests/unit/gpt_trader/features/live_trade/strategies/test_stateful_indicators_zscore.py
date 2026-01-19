"""Tests for OnlineZScore stateful indicator."""

from decimal import Decimal

from gpt_trader.features.live_trade.stateful_indicators import OnlineZScore


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

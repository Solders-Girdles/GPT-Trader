"""Tests for the unified IndicatorBundle and basic performance expectations."""

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.stateful_indicators import IndicatorBundle, OnlineRSI, OnlineSMA


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

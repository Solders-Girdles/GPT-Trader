"""Tests for FundingPnLTracker edge cases and special scenarios."""

from datetime import datetime, timedelta
from decimal import Decimal

from gpt_trader.backtesting.simulation.funding_tracker import FundingPnLTracker


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_small_funding_rate(self) -> None:
        """Test with very small funding rate."""
        tracker = FundingPnLTracker()
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0000001"),  # Very small
            current_time=base_time,
        )

        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0000001"),
            current_time=base_time + timedelta(hours=1),
        )

        # 1 * 40000 * (0.0000001 / 8) = 0.0005
        assert result == Decimal("0.0005")

    def test_very_large_position(self) -> None:
        """Test with very large position size."""
        tracker = FundingPnLTracker()
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1000"),  # 1000 BTC
            mark_price=Decimal("50000"),
            funding_rate_8h=Decimal("0.0001"),
            current_time=base_time,
        )

        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1000"),
            mark_price=Decimal("50000"),
            funding_rate_8h=Decimal("0.0001"),
            current_time=base_time + timedelta(hours=1),
        )

        # 1000 * 50000 * (0.0001 / 8) = 625
        assert result == Decimal("625")

    def test_changing_position_size_between_accruals(self) -> None:
        """Test accrual with changing position size."""
        tracker = FundingPnLTracker()
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time,
        )

        # Increased position
        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("2"),  # Doubled position
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time + timedelta(hours=1),
        )

        # 2 * 40000 * (0.0008 / 8) = 8
        assert result == Decimal("8")

    def test_changing_mark_price_between_accruals(self) -> None:
        """Test accrual with changing mark price."""
        tracker = FundingPnLTracker()
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time,
        )

        # Price increased
        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("50000"),  # Price went up
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time + timedelta(hours=1),
        )

        # 1 * 50000 * (0.0008 / 8) = 5
        assert result == Decimal("5")

    def test_changing_funding_rate_between_accruals(self) -> None:
        """Test accrual with changing funding rate."""
        tracker = FundingPnLTracker()
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time,
        )

        # Funding rate changed
        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("-0.0004"),  # Rate went negative
            current_time=base_time + timedelta(hours=1),
        )

        # 1 * 40000 * (-0.0004 / 8) = -2
        assert result == Decimal("-2")

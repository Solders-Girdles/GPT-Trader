"""Tests for FundingPnLTracker accrual with custom intervals."""

from datetime import datetime, timedelta
from decimal import Decimal

from gpt_trader.backtesting.simulation.funding_tracker import FundingPnLTracker


class TestFundingAccrualCustomIntervals:
    """Test accrual with custom intervals."""

    def test_custom_accrual_interval_4_hours(self) -> None:
        """Test accrual with 4-hour interval."""
        tracker = FundingPnLTracker(accrual_interval_hours=4)
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        # Initialize
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time,
        )

        # Try at 2 hours - should return 0
        result_2h = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time + timedelta(hours=2),
        )
        assert result_2h == Decimal("0")

        # At 4 hours - should calculate
        # Expected: 1 * 40000 * (0.0008 / 8) * 4 = 16
        result_4h = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time + timedelta(hours=4),
        )
        assert result_4h == Decimal("16")

"""Tests for FundingPnLTracker funding accrual timing."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.funding_tracker import FundingPnLTracker


class TestFundingAccrualTiming:
    """Test timing-related funding accrual behavior."""

    @pytest.fixture
    def tracker(self) -> FundingPnLTracker:
        """Create a default funding tracker."""
        return FundingPnLTracker()

    @pytest.fixture
    def base_time(self) -> datetime:
        """Base timestamp for testing."""
        return datetime(2024, 1, 1, 0, 0, 0)

    def test_first_accrual_initializes_and_returns_zero(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that first accrual initializes tracking and returns zero."""
        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("50000"),
            funding_rate_8h=Decimal("0.0001"),
            current_time=base_time,
        )

        assert result == Decimal("0")
        assert tracker.get_accrued("BTC-PERP-USDC") == Decimal("0")

    def test_accrual_before_interval_returns_zero(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that accrual before interval elapsed returns zero."""
        # Initialize
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("50000"),
            funding_rate_8h=Decimal("0.0001"),
            current_time=base_time,
        )

        # Try to accrue after only 30 minutes (less than 1 hour default)
        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("50000"),
            funding_rate_8h=Decimal("0.0001"),
            current_time=base_time + timedelta(minutes=30),
        )

        assert result == Decimal("0")

    def test_accrual_after_interval_calculates_funding(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that accrual after interval calculates correct funding."""
        # Initialize
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("50000"),
            funding_rate_8h=Decimal("0.0008"),  # 0.08% per 8 hours
            current_time=base_time,
        )

        # Accrue after 1 hour
        # Expected: 1 * 50000 * (0.0008 / 8) * 1 = 5
        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("50000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time + timedelta(hours=1),
        )

        assert result == Decimal("5")

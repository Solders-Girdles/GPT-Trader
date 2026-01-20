"""Tests for FundingPnLTracker initialization and basic timing."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.funding_tracker import FundingPnLTracker


class TestFundingPnLTrackerInitialization:
    """Test FundingPnLTracker initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization parameters."""
        tracker = FundingPnLTracker()

        assert tracker.accrual_interval_hours == 1
        assert tracker.settlement_interval_hours == 8
        assert tracker.get_total_funding_pnl() == Decimal("0")

    def test_custom_accrual_interval(self) -> None:
        """Test initialization with custom accrual interval."""
        tracker = FundingPnLTracker(accrual_interval_hours=4)

        assert tracker.accrual_interval_hours == 4
        assert tracker.settlement_interval_hours == 8

    def test_custom_settlement_interval(self) -> None:
        """Test initialization with custom settlement interval."""
        tracker = FundingPnLTracker(settlement_interval_hours=24)

        assert tracker.accrual_interval_hours == 1
        assert tracker.settlement_interval_hours == 24

    def test_both_custom_intervals(self) -> None:
        """Test initialization with both custom intervals."""
        tracker = FundingPnLTracker(
            accrual_interval_hours=2,
            settlement_interval_hours=12,
        )

        assert tracker.accrual_interval_hours == 2
        assert tracker.settlement_interval_hours == 12


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

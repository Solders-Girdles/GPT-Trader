"""Tests for FundingPnLTracker initialization."""

from decimal import Decimal

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

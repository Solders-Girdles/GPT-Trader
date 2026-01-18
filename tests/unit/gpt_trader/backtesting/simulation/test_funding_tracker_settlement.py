"""Tests for FundingPnLTracker settlement behavior."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.funding_tracker import FundingPnLTracker


class TestFundingSettlement:
    """Test funding settlement logic."""

    @pytest.fixture
    def tracker(self) -> FundingPnLTracker:
        """Create a default funding tracker."""
        return FundingPnLTracker()

    @pytest.fixture
    def base_time(self) -> datetime:
        """Base timestamp for testing."""
        return datetime(2024, 1, 1, 0, 0, 0)

    def test_settle_unknown_symbol_returns_zero(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that settling unknown symbol returns zero."""
        result = tracker.settle("UNKNOWN-PERP-USDC", base_time)
        assert result == Decimal("0")

    def test_settle_resets_accrued_funding(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that settlement resets accrued funding."""
        # Initialize and accrue
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time,
        )

        for i in range(1, 5):
            tracker.accrue(
                symbol="BTC-PERP-USDC",
                position_size=Decimal("1"),
                mark_price=Decimal("40000"),
                funding_rate_8h=Decimal("0.0008"),
                current_time=base_time + timedelta(hours=i),
            )

        # Verify accrued is non-zero
        assert tracker.get_accrued("BTC-PERP-USDC") > 0

        # Settle
        settled = tracker.settle("BTC-PERP-USDC", base_time + timedelta(hours=8))

        # Accrued should be reset to zero
        assert tracker.get_accrued("BTC-PERP-USDC") == Decimal("0")
        # Settled amount should equal what was accrued
        assert settled == Decimal("16")  # 4 accruals * 4 each = 16

    def test_settle_before_interval_returns_zero(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that settlement before interval returns zero."""
        # Initialize and accrue
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time,
        )

        for i in range(1, 5):
            tracker.accrue(
                symbol="BTC-PERP-USDC",
                position_size=Decimal("1"),
                mark_price=Decimal("40000"),
                funding_rate_8h=Decimal("0.0008"),
                current_time=base_time + timedelta(hours=i),
            )

        # First settlement works
        first_settle = tracker.settle("BTC-PERP-USDC", base_time + timedelta(hours=5))
        assert first_settle == Decimal("16")

        # Add more accruals
        for i in range(6, 9):
            tracker.accrue(
                symbol="BTC-PERP-USDC",
                position_size=Decimal("1"),
                mark_price=Decimal("40000"),
                funding_rate_8h=Decimal("0.0008"),
                current_time=base_time + timedelta(hours=i),
            )

        # Try to settle again too soon (only 5 hours after first settlement)
        second_settle = tracker.settle("BTC-PERP-USDC", base_time + timedelta(hours=10))
        assert second_settle == Decimal("0")  # Not enough time passed

    def test_settle_updates_total_paid(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that settlement updates total paid."""
        # Initialize and accrue
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time,
        )

        for i in range(1, 5):
            tracker.accrue(
                symbol="BTC-PERP-USDC",
                position_size=Decimal("1"),
                mark_price=Decimal("40000"),
                funding_rate_8h=Decimal("0.0008"),
                current_time=base_time + timedelta(hours=i),
            )

        # Before settlement
        assert tracker.get_total_paid("BTC-PERP-USDC") == Decimal("0")

        # Settle
        tracker.settle("BTC-PERP-USDC", base_time + timedelta(hours=8))

        # After settlement
        assert tracker.get_total_paid("BTC-PERP-USDC") == Decimal("16")

    def test_settle_creates_funding_event(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that settlement creates a funding event."""
        # Initialize and accrue
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time,
        )

        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time + timedelta(hours=1),
        )

        # Before settlement - no events
        assert len(tracker.get_funding_events()) == 0

        # Settle
        settle_time = base_time + timedelta(hours=8)
        tracker.settle("BTC-PERP-USDC", settle_time)

        # After settlement - one event
        events = tracker.get_funding_events()
        assert len(events) == 1
        assert events[0].symbol == "BTC-PERP-USDC"
        assert events[0].timestamp == settle_time
        assert events[0].amount == Decimal("4")


class TestFundingSettlementCustomInterval:
    """Test settlement with custom interval."""

    def test_custom_settlement_interval_24_hours(self) -> None:
        """Test settlement with 24-hour interval."""
        tracker = FundingPnLTracker(settlement_interval_hours=24)
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        # Initialize and accrue
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time,
        )

        for i in range(1, 9):
            tracker.accrue(
                symbol="BTC-PERP-USDC",
                position_size=Decimal("1"),
                mark_price=Decimal("40000"),
                funding_rate_8h=Decimal("0.0008"),
                current_time=base_time + timedelta(hours=i),
            )

        # First settlement
        first = tracker.settle("BTC-PERP-USDC", base_time + timedelta(hours=12))
        assert first == Decimal("32")  # 8 * 4 = 32

        # More accruals
        for i in range(13, 21):
            tracker.accrue(
                symbol="BTC-PERP-USDC",
                position_size=Decimal("1"),
                mark_price=Decimal("40000"),
                funding_rate_8h=Decimal("0.0008"),
                current_time=base_time + timedelta(hours=i),
            )

        # Try at 20 hours (only 8 hours since first settlement) - should fail
        result = tracker.settle("BTC-PERP-USDC", base_time + timedelta(hours=20))
        assert result == Decimal("0")

        # At 36 hours (24 hours after first settlement) - should work
        result = tracker.settle("BTC-PERP-USDC", base_time + timedelta(hours=36))
        assert result == Decimal("32")  # 8 * 4 = 32

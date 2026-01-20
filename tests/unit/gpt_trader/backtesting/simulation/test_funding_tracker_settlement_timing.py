"""Tests for FundingPnLTracker settlement timing and should_settle behavior."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.funding_tracker import FundingPnLTracker


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


class TestShouldSettle:
    """Test should_settle method."""

    @pytest.fixture
    def tracker(self) -> FundingPnLTracker:
        """Create a default funding tracker."""
        return FundingPnLTracker()

    @pytest.fixture
    def base_time(self) -> datetime:
        """Base timestamp for testing."""
        return datetime(2024, 1, 1, 0, 0, 0)

    def test_should_settle_first_time(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test should_settle returns True for first settlement."""
        assert tracker.should_settle(base_time, "BTC-PERP-USDC") is True

    def test_should_settle_after_interval(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test should_settle returns True after interval elapsed."""
        # Initialize
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time,
        )

        # First settlement
        tracker.settle("BTC-PERP-USDC", base_time)

        # After 8 hours
        assert tracker.should_settle(base_time + timedelta(hours=8), "BTC-PERP-USDC") is True

    def test_should_settle_before_interval(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test should_settle returns False before interval elapsed."""
        # Initialize
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time,
        )

        # First settlement
        tracker.settle("BTC-PERP-USDC", base_time)

        # After only 4 hours
        assert tracker.should_settle(base_time + timedelta(hours=4), "BTC-PERP-USDC") is False

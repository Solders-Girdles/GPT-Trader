"""Tests for FundingPnLTracker accrual sign and accumulation behavior."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.funding_tracker import FundingPnLTracker


class TestFundingAccrualSignsAndAccumulation:
    """Test funding accrual across long/short and accumulation scenarios."""

    @pytest.fixture
    def tracker(self) -> FundingPnLTracker:
        """Create a default funding tracker."""
        return FundingPnLTracker()

    @pytest.fixture
    def base_time(self) -> datetime:
        """Base timestamp for testing."""
        return datetime(2024, 1, 1, 0, 0, 0)

    def test_long_position_positive_rate_pays_funding(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that long position with positive rate pays funding."""
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("2"),  # Long 2 BTC
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0001"),
            current_time=base_time,
        )

        # Expected: 2 * 40000 * (0.0001 / 8) * 1 = 1
        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("2"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0001"),
            current_time=base_time + timedelta(hours=1),
        )

        assert result == Decimal("1")
        assert result > 0  # Positive = paid

    def test_short_position_positive_rate_receives_funding(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that short position with positive rate receives funding."""
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("-2"),  # Short 2 BTC
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0001"),
            current_time=base_time,
        )

        # Expected: -2 * 40000 * (0.0001 / 8) * 1 = -1
        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("-2"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0001"),
            current_time=base_time + timedelta(hours=1),
        )

        assert result == Decimal("-1")
        assert result < 0  # Negative = received

    def test_long_position_negative_rate_receives_funding(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that long position with negative rate receives funding."""
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("2"),  # Long 2 BTC
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("-0.0001"),  # Negative rate
            current_time=base_time,
        )

        # Expected: 2 * 40000 * (-0.0001 / 8) * 1 = -1
        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("2"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("-0.0001"),
            current_time=base_time + timedelta(hours=1),
        )

        assert result == Decimal("-1")
        assert result < 0  # Negative = received

    def test_short_position_negative_rate_pays_funding(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that short position with negative rate pays funding."""
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("-2"),  # Short 2 BTC
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("-0.0001"),  # Negative rate
            current_time=base_time,
        )

        # Expected: -2 * 40000 * (-0.0001 / 8) * 1 = 1
        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("-2"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("-0.0001"),
            current_time=base_time + timedelta(hours=1),
        )

        assert result == Decimal("1")
        assert result > 0  # Positive = paid

    def test_multiple_accruals_accumulate(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that multiple accruals accumulate."""
        # Initialize
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time,
        )

        # Accrue 3 times
        for i in range(1, 4):
            tracker.accrue(
                symbol="BTC-PERP-USDC",
                position_size=Decimal("1"),
                mark_price=Decimal("40000"),
                funding_rate_8h=Decimal("0.0008"),
                current_time=base_time + timedelta(hours=i),
            )

        # Each accrual = 1 * 40000 * (0.0008 / 8) * 1 = 4
        # Total accrued = 3 * 4 = 12
        assert tracker.get_accrued("BTC-PERP-USDC") == Decimal("12")

    def test_zero_position_size_no_funding(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that zero position size results in no funding."""
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("0"),
            mark_price=Decimal("50000"),
            funding_rate_8h=Decimal("0.0001"),
            current_time=base_time,
        )

        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("0"),
            mark_price=Decimal("50000"),
            funding_rate_8h=Decimal("0.0001"),
            current_time=base_time + timedelta(hours=1),
        )

        assert result == Decimal("0")

    def test_zero_funding_rate_no_funding(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that zero funding rate results in no funding."""
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("50000"),
            funding_rate_8h=Decimal("0"),
            current_time=base_time,
        )

        result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("50000"),
            funding_rate_8h=Decimal("0"),
            current_time=base_time + timedelta(hours=1),
        )

        assert result == Decimal("0")

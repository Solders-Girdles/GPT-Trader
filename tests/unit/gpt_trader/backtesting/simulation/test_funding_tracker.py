"""Comprehensive tests for FundingPnLTracker."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.funding_tracker import (
    FundingEvent,
    FundingPnLTracker,
)


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


class TestFundingAccrual:
    """Test funding accrual logic."""

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


class TestMultiSymbolTracking:
    """Test tracking across multiple symbols."""

    @pytest.fixture
    def tracker(self) -> FundingPnLTracker:
        """Create a default funding tracker."""
        return FundingPnLTracker()

    @pytest.fixture
    def base_time(self) -> datetime:
        """Base timestamp for testing."""
        return datetime(2024, 1, 1, 0, 0, 0)

    def test_independent_symbol_tracking(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test that symbols are tracked independently."""
        # Initialize both
        tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time,
        )

        tracker.accrue(
            symbol="ETH-PERP-USDC",
            position_size=Decimal("10"),
            mark_price=Decimal("2000"),
            funding_rate_8h=Decimal("0.0004"),
            current_time=base_time,
        )

        # Accrue for BTC
        btc_result = tracker.accrue(
            symbol="BTC-PERP-USDC",
            position_size=Decimal("1"),
            mark_price=Decimal("40000"),
            funding_rate_8h=Decimal("0.0008"),
            current_time=base_time + timedelta(hours=1),
        )

        # Accrue for ETH
        eth_result = tracker.accrue(
            symbol="ETH-PERP-USDC",
            position_size=Decimal("10"),
            mark_price=Decimal("2000"),
            funding_rate_8h=Decimal("0.0004"),
            current_time=base_time + timedelta(hours=1),
        )

        # BTC: 1 * 40000 * (0.0008/8) = 4
        assert btc_result == Decimal("4")
        # ETH: 10 * 2000 * (0.0004/8) = 1
        assert eth_result == Decimal("1")

        assert tracker.get_accrued("BTC-PERP-USDC") == Decimal("4")
        assert tracker.get_accrued("ETH-PERP-USDC") == Decimal("1")

    def test_independent_settlement(self, tracker: FundingPnLTracker, base_time: datetime) -> None:
        """Test that settlements are independent."""
        # Initialize and accrue both
        for symbol, size, price, rate in [
            ("BTC-PERP-USDC", Decimal("1"), Decimal("40000"), Decimal("0.0008")),
            ("ETH-PERP-USDC", Decimal("10"), Decimal("2000"), Decimal("0.0004")),
        ]:
            tracker.accrue(
                symbol=symbol,
                position_size=size,
                mark_price=price,
                funding_rate_8h=rate,
                current_time=base_time,
            )
            tracker.accrue(
                symbol=symbol,
                position_size=size,
                mark_price=price,
                funding_rate_8h=rate,
                current_time=base_time + timedelta(hours=1),
            )

        # Settle BTC only
        btc_settled = tracker.settle("BTC-PERP-USDC", base_time + timedelta(hours=8))

        assert btc_settled == Decimal("4")
        assert tracker.get_accrued("BTC-PERP-USDC") == Decimal("0")
        assert tracker.get_accrued("ETH-PERP-USDC") == Decimal("1")  # Unchanged

    def test_total_funding_pnl_across_symbols(
        self, tracker: FundingPnLTracker, base_time: datetime
    ) -> None:
        """Test total funding PnL calculation across symbols."""
        # Initialize and accrue both symbols
        for symbol, size, price, rate in [
            ("BTC-PERP-USDC", Decimal("1"), Decimal("40000"), Decimal("0.0008")),
            ("ETH-PERP-USDC", Decimal("-10"), Decimal("2000"), Decimal("0.0004")),
        ]:
            tracker.accrue(
                symbol=symbol,
                position_size=size,
                mark_price=price,
                funding_rate_8h=rate,
                current_time=base_time,
            )
            tracker.accrue(
                symbol=symbol,
                position_size=size,
                mark_price=price,
                funding_rate_8h=rate,
                current_time=base_time + timedelta(hours=1),
            )

        # Settle both
        tracker.settle("BTC-PERP-USDC", base_time + timedelta(hours=8))
        tracker.settle("ETH-PERP-USDC", base_time + timedelta(hours=8))

        # BTC: 4 paid, ETH: -1 received
        # Total: 4 + (-1) = 3
        assert tracker.get_total_paid("BTC-PERP-USDC") == Decimal("4")
        assert tracker.get_total_paid("ETH-PERP-USDC") == Decimal("-1")
        assert tracker.get_total_funding_pnl() == Decimal("3")


class TestGetAccruedAndTotalPaid:
    """Test get_accrued and get_total_paid methods."""

    def test_get_accrued_unknown_symbol(self) -> None:
        """Test get_accrued returns zero for unknown symbol."""
        tracker = FundingPnLTracker()
        assert tracker.get_accrued("UNKNOWN-PERP-USDC") == Decimal("0")

    def test_get_total_paid_unknown_symbol(self) -> None:
        """Test get_total_paid returns zero for unknown symbol."""
        tracker = FundingPnLTracker()
        assert tracker.get_total_paid("UNKNOWN-PERP-USDC") == Decimal("0")

    def test_get_total_funding_pnl_empty(self) -> None:
        """Test get_total_funding_pnl returns zero when no symbols tracked."""
        tracker = FundingPnLTracker()
        assert tracker.get_total_funding_pnl() == Decimal("0")


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


class TestFundingEventFiltering:
    """Test funding event filtering."""

    @pytest.fixture
    def populated_tracker(self) -> FundingPnLTracker:
        """Create a tracker with multiple events."""
        tracker = FundingPnLTracker()
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        # Create events for BTC at hours 8, 16, 24
        for hour in [0, 1, 2, 3, 4, 5, 6, 7]:
            tracker.accrue(
                symbol="BTC-PERP-USDC",
                position_size=Decimal("1"),
                mark_price=Decimal("40000"),
                funding_rate_8h=Decimal("0.0008"),
                current_time=base_time + timedelta(hours=hour),
            )

        tracker.settle("BTC-PERP-USDC", base_time + timedelta(hours=8))

        for hour in [9, 10, 11, 12, 13, 14, 15]:
            tracker.accrue(
                symbol="BTC-PERP-USDC",
                position_size=Decimal("1"),
                mark_price=Decimal("40000"),
                funding_rate_8h=Decimal("0.0008"),
                current_time=base_time + timedelta(hours=hour),
            )

        tracker.settle("BTC-PERP-USDC", base_time + timedelta(hours=16))

        # Create events for ETH at hour 8
        for hour in [0, 1, 2, 3, 4, 5, 6, 7]:
            tracker.accrue(
                symbol="ETH-PERP-USDC",
                position_size=Decimal("10"),
                mark_price=Decimal("2000"),
                funding_rate_8h=Decimal("0.0004"),
                current_time=base_time + timedelta(hours=hour),
            )

        tracker.settle("ETH-PERP-USDC", base_time + timedelta(hours=8))

        return tracker

    def test_get_all_events(self, populated_tracker: FundingPnLTracker) -> None:
        """Test getting all events without filters."""
        events = populated_tracker.get_funding_events()
        assert len(events) == 3  # 2 BTC + 1 ETH

    def test_filter_by_symbol(self, populated_tracker: FundingPnLTracker) -> None:
        """Test filtering events by symbol."""
        btc_events = populated_tracker.get_funding_events(symbol="BTC-PERP-USDC")
        assert len(btc_events) == 2
        assert all(e.symbol == "BTC-PERP-USDC" for e in btc_events)

        eth_events = populated_tracker.get_funding_events(symbol="ETH-PERP-USDC")
        assert len(eth_events) == 1
        assert all(e.symbol == "ETH-PERP-USDC" for e in eth_events)

    def test_filter_by_start_time(self, populated_tracker: FundingPnLTracker) -> None:
        """Test filtering events by start time."""
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        # Only events at hour 16 and later
        events = populated_tracker.get_funding_events(start=base_time + timedelta(hours=10))
        assert len(events) == 1
        assert events[0].timestamp == base_time + timedelta(hours=16)

    def test_filter_by_end_time(self, populated_tracker: FundingPnLTracker) -> None:
        """Test filtering events by end time."""
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        # Only events before hour 10
        events = populated_tracker.get_funding_events(end=base_time + timedelta(hours=10))
        assert len(events) == 2  # Both at hour 8

    def test_filter_by_time_range(self, populated_tracker: FundingPnLTracker) -> None:
        """Test filtering events by time range."""
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        # Only events between hours 5 and 12
        events = populated_tracker.get_funding_events(
            start=base_time + timedelta(hours=5),
            end=base_time + timedelta(hours=12),
        )
        assert len(events) == 2  # Both hour 8 events

    def test_filter_by_symbol_and_time(self, populated_tracker: FundingPnLTracker) -> None:
        """Test filtering events by symbol and time range."""
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        events = populated_tracker.get_funding_events(
            symbol="BTC-PERP-USDC",
            start=base_time + timedelta(hours=10),
        )
        assert len(events) == 1
        assert events[0].symbol == "BTC-PERP-USDC"

    def test_filter_returns_empty_for_no_matches(
        self, populated_tracker: FundingPnLTracker
    ) -> None:
        """Test filtering returns empty list when no events match."""
        events = populated_tracker.get_funding_events(symbol="NONEXISTENT-PERP-USDC")
        assert events == []


class TestFundingEvent:
    """Test FundingEvent class."""

    def test_funding_event_creation(self) -> None:
        """Test FundingEvent creation."""
        timestamp = datetime(2024, 1, 1, 8, 0, 0)
        event = FundingEvent(
            symbol="BTC-PERP-USDC",
            timestamp=timestamp,
            amount=Decimal("5.25"),
        )

        assert event.symbol == "BTC-PERP-USDC"
        assert event.timestamp == timestamp
        assert event.amount == Decimal("5.25")

    def test_funding_event_repr_paid(self) -> None:
        """Test FundingEvent repr for paid funding."""
        event = FundingEvent(
            symbol="BTC-PERP-USDC",
            timestamp=datetime(2024, 1, 1, 8, 0, 0),
            amount=Decimal("5.25"),
        )

        repr_str = repr(event)
        assert "BTC-PERP-USDC" in repr_str
        assert "paid" in repr_str
        assert "5.25" in repr_str

    def test_funding_event_repr_received(self) -> None:
        """Test FundingEvent repr for received funding."""
        event = FundingEvent(
            symbol="ETH-PERP-USDC",
            timestamp=datetime(2024, 1, 1, 8, 0, 0),
            amount=Decimal("-3.50"),
        )

        repr_str = repr(event)
        assert "ETH-PERP-USDC" in repr_str
        assert "received" in repr_str
        assert "3.50" in repr_str


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

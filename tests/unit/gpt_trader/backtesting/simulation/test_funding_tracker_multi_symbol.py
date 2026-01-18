"""Tests for FundingPnLTracker multi-symbol tracking and totals."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.funding_tracker import FundingPnLTracker


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

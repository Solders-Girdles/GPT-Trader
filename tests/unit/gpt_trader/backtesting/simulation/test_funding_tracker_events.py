"""Tests for FundingPnLTracker funding events and filtering."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.funding_tracker import (
    FundingEvent,
    FundingPnLTracker,
)


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

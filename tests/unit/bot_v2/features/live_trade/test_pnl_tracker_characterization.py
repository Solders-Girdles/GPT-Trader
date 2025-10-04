"""
Characterization tests for PnLTracker.

These tests lock in the current orchestration and integration behavior
of PnLTracker before extraction. They complement the existing unit tests
by focusing on end-to-end workflows and state management.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.features.live_trade.pnl_tracker import (
    FundingCalculator,
    PnLTracker,
    PositionState,
)


# ============================================================================
# Position State Lifecycle
# ============================================================================


class TestPositionStateLifecycle:
    """Test position state transitions and lifecycle."""

    def test_new_position_starts_with_zero_state(self):
        """New position has zero PnL and empty state."""
        position = PositionState(symbol="BTC-PERP")

        assert position.symbol == "BTC-PERP"
        assert position.side is None
        assert position.quantity == Decimal("0")
        assert position.avg_entry_price == Decimal("0")
        assert position.realized_pnl == Decimal("0")
        assert position.unrealized_pnl == Decimal("0")
        assert position.funding_paid == Decimal("0")
        assert position.trades_count == 0
        assert position.winning_trades == 0
        assert position.losing_trades == 0

    def test_opening_long_position_sets_state(self):
        """Opening long position sets side, quantity, entry price."""
        position = PositionState(symbol="BTC-PERP")

        result = position.update_position("buy", Decimal("0.1"), Decimal("50000"))

        assert position.side == "long"
        assert position.quantity == Decimal("0.1")
        assert position.avg_entry_price == Decimal("50000")
        assert position.trades_count == 1
        assert result["realized_pnl"] == Decimal("0")

    def test_opening_short_position_sets_state(self):
        """Opening short position sets side, quantity, entry price."""
        position = PositionState(symbol="ETH-PERP")

        result = position.update_position("sell", Decimal("1.0"), Decimal("3000"))

        assert position.side == "short"
        assert position.quantity == Decimal("1.0")
        assert position.avg_entry_price == Decimal("3000")
        assert position.trades_count == 1
        assert result["realized_pnl"] == Decimal("0")

    def test_adding_to_long_position_updates_weighted_average(self):
        """Adding to position updates weighted average entry price."""
        position = PositionState(symbol="BTC-PERP")
        position.update_position("buy", Decimal("0.1"), Decimal("50000"))

        # Add to position at different price
        position.update_position("buy", Decimal("0.1"), Decimal("52000"))

        assert position.quantity == Decimal("0.2")
        assert position.avg_entry_price == Decimal("51000")  # (50000 + 52000) / 2
        assert position.side == "long"

    def test_closing_long_position_realizes_pnl(self):
        """Closing position realizes PnL and resets state."""
        position = PositionState(symbol="BTC-PERP")
        position.update_position("buy", Decimal("0.1"), Decimal("50000"))

        # Close at profit
        result = position.update_position("sell", Decimal("0.1"), Decimal("52000"), is_reduce=True)

        assert result["realized_pnl"] == Decimal("200")  # (52000 - 50000) * 0.1
        assert position.realized_pnl == Decimal("200")
        assert position.quantity == Decimal("0")
        assert position.side is None
        assert position.avg_entry_price == Decimal("0")
        assert position.winning_trades == 1

    def test_partial_close_maintains_position(self):
        """Partial close realizes proportional PnL, maintains remainder."""
        position = PositionState(symbol="BTC-PERP")
        position.update_position("buy", Decimal("1.0"), Decimal("50000"))

        # Close 30%
        result = position.update_position("sell", Decimal("0.3"), Decimal("51000"), is_reduce=True)

        assert result["realized_pnl"] == Decimal("300")  # (51000 - 50000) * 0.3
        assert position.realized_pnl == Decimal("300")
        assert position.quantity == Decimal("0.7")
        assert position.side == "long"
        assert position.avg_entry_price == Decimal("50000")  # Unchanged

    def test_flipping_position_long_to_short(self):
        """Over-closing flips position and tracks realized PnL."""
        position = PositionState(symbol="BTC-PERP")
        position.update_position("buy", Decimal("0.1"), Decimal("50000"))

        # Sell more than we have (0.2 > 0.1)
        result = position.update_position("sell", Decimal("0.2"), Decimal("51000"))

        # Realized PnL from closing 0.1 long
        assert result["realized_pnl"] == Decimal("100")  # (51000 - 50000) * 0.1
        assert position.realized_pnl == Decimal("100")

        # Now short 0.1
        assert position.side == "short"
        assert position.quantity == Decimal("0.1")
        assert position.avg_entry_price == Decimal("51000")
        assert position.trades_count == 2

    def test_update_mark_calculates_unrealized_pnl(self):
        """update_mark calculates unrealized PnL for open position."""
        position = PositionState(symbol="BTC-PERP")
        position.update_position("buy", Decimal("0.1"), Decimal("50000"))

        unrealized = position.update_mark(Decimal("52000"))

        assert unrealized == Decimal("200")  # (52000 - 50000) * 0.1
        assert position.unrealized_pnl == Decimal("200")

    def test_update_mark_tracks_drawdown(self):
        """update_mark tracks peak equity and max drawdown."""
        position = PositionState(symbol="BTC-PERP")
        position.update_position("buy", Decimal("0.1"), Decimal("50000"))

        # Profit
        position.update_mark(Decimal("52000"))
        assert position.peak_equity == Decimal("200")
        assert position.max_drawdown == Decimal("0")

        # Drawdown
        position.update_mark(Decimal("50500"))
        assert position.unrealized_pnl == Decimal("50")
        assert position.max_drawdown == Decimal("150")  # 200 - 50

    def test_zero_position_has_no_unrealized_pnl(self):
        """Zero position always has zero unrealized PnL."""
        position = PositionState(symbol="BTC-PERP")

        unrealized = position.update_mark(Decimal("60000"))

        assert unrealized == Decimal("0")
        assert position.unrealized_pnl == Decimal("0")


# ============================================================================
# PnLTracker Orchestration
# ============================================================================


class TestPnLTrackerOrchestration:
    """Test PnLTracker's orchestration of multiple positions."""

    def test_get_or_create_position_creates_new(self):
        """get_or_create_position creates new position if not exists."""
        tracker = PnLTracker()

        position = tracker.get_or_create_position("BTC-PERP")

        assert position.symbol == "BTC-PERP"
        assert "BTC-PERP" in tracker.positions

    def test_get_or_create_position_returns_existing(self):
        """get_or_create_position returns existing position."""
        tracker = PnLTracker()
        position1 = tracker.get_or_create_position("BTC-PERP")
        position1.realized_pnl = Decimal("100")

        position2 = tracker.get_or_create_position("BTC-PERP")

        assert position2 is position1
        assert position2.realized_pnl == Decimal("100")

    def test_update_position_delegates_to_position_state(self):
        """update_position delegates to PositionState."""
        tracker = PnLTracker()

        result = tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))

        assert result["realized_pnl"] == Decimal("0")
        assert tracker.positions["BTC-PERP"].quantity == Decimal("0.1")

    def test_update_marks_processes_all_positions(self):
        """update_marks updates all positions with current marks."""
        tracker = PnLTracker()
        tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))
        tracker.update_position("ETH-PERP", "buy", Decimal("1.0"), Decimal("3000"))

        unrealized = tracker.update_marks(
            {
                "BTC-PERP": Decimal("52000"),
                "ETH-PERP": Decimal("3200"),
            }
        )

        assert unrealized["BTC-PERP"] == Decimal("200")
        assert unrealized["ETH-PERP"] == Decimal("200")
        assert tracker.positions["BTC-PERP"].unrealized_pnl == Decimal("200")
        assert tracker.positions["ETH-PERP"].unrealized_pnl == Decimal("200")

    def test_update_marks_skips_unknown_symbols(self):
        """update_marks only processes positions that exist."""
        tracker = PnLTracker()
        tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))

        unrealized = tracker.update_marks(
            {
                "BTC-PERP": Decimal("51000"),
                "UNKNOWN-PERP": Decimal("1000"),  # Not tracked
            }
        )

        assert "BTC-PERP" in unrealized
        assert "UNKNOWN-PERP" not in unrealized

    def test_get_total_pnl_aggregates_all_positions(self):
        """get_total_pnl aggregates realized/unrealized across positions."""
        tracker = PnLTracker()

        # BTC: +200 unrealized
        tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))
        tracker.update_marks({"BTC-PERP": Decimal("52000")})

        # ETH: +100 realized, +50 unrealized
        tracker.update_position("ETH-PERP", "buy", Decimal("1.0"), Decimal("3000"))
        tracker.update_position("ETH-PERP", "sell", Decimal("0.5"), Decimal("3200"), is_reduce=True)
        tracker.update_marks({"ETH-PERP": Decimal("3100")})

        total = tracker.get_total_pnl()

        assert total["realized"] == Decimal("100")  # ETH only
        assert total["unrealized"] == Decimal("250")  # BTC (200) + ETH (50)
        assert total["total"] == Decimal("350")

    def test_get_total_pnl_includes_funding(self):
        """get_total_pnl includes funding paid."""
        tracker = PnLTracker()
        tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))
        tracker.positions["BTC-PERP"].funding_paid = Decimal("5")

        total = tracker.get_total_pnl()

        assert total["funding"] == Decimal("5")

    def test_multiple_symbols_tracked_independently(self):
        """Multiple symbols tracked with independent state."""
        tracker = PnLTracker()

        # BTC long
        tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))

        # ETH short
        tracker.update_position("ETH-PERP", "sell", Decimal("1.0"), Decimal("3000"))

        assert tracker.positions["BTC-PERP"].side == "long"
        assert tracker.positions["ETH-PERP"].side == "short"
        assert len(tracker.positions) == 2


# ============================================================================
# Funding Integration
# ============================================================================


class TestFundingIntegration:
    """Test funding accrual integration."""

    def test_accrue_funding_delegates_to_calculator(self):
        """accrue_funding delegates to FundingCalculator."""
        tracker = PnLTracker()
        tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))

        # Mock last funding time in the past
        tracker.positions["BTC-PERP"].last_funding_time = datetime.now() - timedelta(hours=9)

        funding = tracker.accrue_funding(
            "BTC-PERP",
            Decimal("50000"),
            Decimal("0.01"),  # 1% funding rate
        )

        # Longs pay when rate is positive
        assert funding == Decimal("-50")  # -(0.1 * 50000 * 0.01)
        assert tracker.positions["BTC-PERP"].funding_paid == Decimal("50")

    def test_accrue_funding_updates_last_funding_time(self):
        """accrue_funding updates last_funding_time."""
        tracker = PnLTracker()
        tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))
        tracker.positions["BTC-PERP"].last_funding_time = datetime.now() - timedelta(hours=9)

        before = tracker.positions["BTC-PERP"].last_funding_time
        tracker.accrue_funding("BTC-PERP", Decimal("50000"), Decimal("0.01"))
        after = tracker.positions["BTC-PERP"].last_funding_time

        assert after > before

    def test_accrue_funding_returns_none_if_not_due(self):
        """accrue_funding returns None if funding not due."""
        tracker = PnLTracker()
        tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))

        # Just accrued
        tracker.positions["BTC-PERP"].last_funding_time = datetime.now()

        funding = tracker.accrue_funding("BTC-PERP", Decimal("50000"), Decimal("0.01"))

        assert funding is None

    def test_accrue_funding_respects_next_funding_time(self):
        """accrue_funding uses next_funding_time if provided."""
        tracker = PnLTracker()
        tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))

        # Next funding in the past
        past_time = datetime.now() - timedelta(hours=1)

        funding = tracker.accrue_funding(
            "BTC-PERP",
            Decimal("50000"),
            Decimal("0.01"),
            next_funding_time=past_time,
        )

        assert funding is not None
        assert tracker.positions["BTC-PERP"].funding_paid > 0


# ============================================================================
# Daily Metrics Generation
# ============================================================================


class TestDailyMetricsGeneration:
    """Test daily metrics generation."""

    def test_generate_daily_metrics_initializes_tracking(self):
        """generate_daily_metrics initializes daily tracking on first call."""
        tracker = PnLTracker()

        metrics = tracker.generate_daily_metrics(Decimal("10000"))

        assert tracker.daily_start_equity is not None
        assert tracker.daily_start_time is not None
        assert metrics["equity"] == 10000.0

    def test_generate_daily_metrics_calculates_daily_return(self):
        """generate_daily_metrics calculates daily return."""
        tracker = PnLTracker()
        tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))
        tracker.update_marks({"BTC-PERP": Decimal("52000")})  # +200 unrealized

        # Start equity: 10000, current: 10200
        metrics = tracker.generate_daily_metrics(Decimal("10200"))

        assert metrics["equity"] == 10200.0
        # Daily return should be ~2% (200 / 10000)
        assert abs(metrics["daily_return"] - 0.02) < 0.001

    def test_generate_daily_metrics_aggregates_positions(self):
        """generate_daily_metrics aggregates all position metrics."""
        tracker = PnLTracker()

        # BTC trade
        tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))
        tracker.update_position(
            "BTC-PERP", "sell", Decimal("0.05"), Decimal("51000"), is_reduce=True
        )

        # ETH trade
        tracker.update_position("ETH-PERP", "buy", Decimal("1.0"), Decimal("3000"))

        tracker.update_marks({"BTC-PERP": Decimal("50500"), "ETH-PERP": Decimal("3100")})

        metrics = tracker.generate_daily_metrics(Decimal("10000"))

        assert metrics["trades"] == 2  # 2 position openings (trades_count only increments on opens)
        assert metrics["positions"] == 2  # 2 open positions

    def test_generate_daily_metrics_includes_all_pnl_types(self):
        """generate_daily_metrics includes realized, unrealized, funding."""
        tracker = PnLTracker()
        tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))
        tracker.update_position(
            "BTC-PERP", "sell", Decimal("0.05"), Decimal("51000"), is_reduce=True
        )
        tracker.update_marks({"BTC-PERP": Decimal("50500")})
        tracker.positions["BTC-PERP"].funding_paid = Decimal("5")

        metrics = tracker.generate_daily_metrics(Decimal("10000"))

        assert metrics["realized_pnl"] == 50.0  # (51000 - 50000) * 0.05
        assert metrics["unrealized_pnl"] == 25.0  # (50500 - 50000) * 0.05
        assert metrics["funding_paid"] == 5.0
        assert metrics["total_pnl"] == 75.0  # 50 + 25

    def test_generate_daily_metrics_calculates_win_rate(self):
        """generate_daily_metrics calculates win rate.

        NOTE: Current implementation has a bug - win_rate = winning_trades / trades_count,
        but winning_trades increments on reduces while trades_count only increments on opens.
        This can produce win_rate > 1.0 (nonsensical). Documenting existing behavior.
        """
        tracker = PnLTracker()
        tracker.update_position(
            "BTC-PERP", "buy", Decimal("0.1"), Decimal("50000")
        )  # trades_count=1

        # Winning trade (increments winning_trades)
        tracker.update_position(
            "BTC-PERP", "sell", Decimal("0.05"), Decimal("51000"), is_reduce=True
        )

        # Losing trade (increments losing_trades)
        tracker.update_position(
            "BTC-PERP", "sell", Decimal("0.05"), Decimal("49000"), is_reduce=True
        )

        metrics = tracker.generate_daily_metrics(Decimal("10000"))

        assert metrics["trades"] == 1  # Only 1 position opening
        # win_rate = 1 winning / 1 trades_count = 1.0 (100%)
        # This is buggy but is current behavior
        assert abs(metrics["win_rate"] - 1.0) < 0.001

    def test_generate_daily_metrics_resets_daily_on_new_day(self):
        """generate_daily_metrics resets tracking after 24 hours.

        NOTE: daily_start_time always resets to current day's midnight (00:00:00),
        not a future timestamp. So if called twice on the same day, it resets to
        the same midnight timestamp.
        """
        tracker = PnLTracker()

        # First day
        tracker.generate_daily_metrics(Decimal("10000"))
        first_start_time = tracker.daily_start_time
        first_equity = tracker.daily_start_equity

        # Simulate next day (manually set time >24h in past to trigger reset)
        tracker.daily_start_time = datetime.now() - timedelta(days=1, hours=1)

        # Next day - should trigger reset
        tracker.generate_daily_metrics(Decimal("10500"))

        # daily_start_time resets to today's midnight (same as first call if on same day)
        # daily_start_equity should update based on new equity calculation
        assert tracker.daily_start_time == first_start_time  # Both are today's midnight
        assert tracker.daily_start_equity != first_equity  # Equity recalculated


# ============================================================================
# Position Metrics
# ============================================================================


class TestPositionMetrics:
    """Test position metrics generation."""

    def test_get_position_metrics_returns_all_positions(self):
        """get_position_metrics returns metrics for all positions."""
        tracker = PnLTracker()
        tracker.update_position("BTC-PERP", "buy", Decimal("0.1"), Decimal("50000"))
        tracker.update_position("ETH-PERP", "sell", Decimal("1.0"), Decimal("3000"))

        metrics = tracker.get_position_metrics()

        assert len(metrics) == 2
        symbols = {m["symbol"] for m in metrics}
        assert "BTC-PERP" in symbols
        assert "ETH-PERP" in symbols

    def test_position_metrics_includes_complete_info(self):
        """Position metrics include all relevant fields.

        NOTE: trades_count only increments on position openings, not reduces.
        """
        tracker = PnLTracker()
        tracker.update_position(
            "BTC-PERP", "buy", Decimal("0.1"), Decimal("50000")
        )  # trades_count=1
        tracker.update_position(
            "BTC-PERP", "sell", Decimal("0.05"), Decimal("51000"), is_reduce=True
        )  # trades_count=1
        tracker.update_marks({"BTC-PERP": Decimal("50500")})
        tracker.positions["BTC-PERP"].funding_paid = Decimal("5")

        metrics = tracker.get_position_metrics()[0]

        assert metrics["symbol"] == "BTC-PERP"
        assert metrics["side"] == "long"
        assert metrics["quantity"] == 0.05
        assert metrics["avg_entry"] == 50000.0
        assert metrics["realized_pnl"] == 50.0  # (51000 - 50000) * 0.05
        assert metrics["unrealized_pnl"] == 25.0  # (50500 - 50000) * 0.05
        assert metrics["total_pnl"] == 75.0
        assert metrics["funding_paid"] == 5.0
        assert metrics["trades"] == 1  # Only 1 position opening


# ============================================================================
# FundingCalculator Standalone
# ============================================================================


class TestFundingCalculator:
    """Test FundingCalculator standalone behavior."""

    def test_calculate_funding_long_positive_rate(self):
        """Longs pay when rate is positive."""
        calculator = FundingCalculator()

        funding = calculator.calculate_funding(
            position_size=Decimal("0.1"),
            side="long",
            mark_price=Decimal("50000"),
            funding_rate=Decimal("0.01"),
        )

        # -(0.1 * 50000 * 0.01) = -50
        assert funding == Decimal("-50")

    def test_calculate_funding_short_positive_rate(self):
        """Shorts receive when rate is positive."""
        calculator = FundingCalculator()

        funding = calculator.calculate_funding(
            position_size=Decimal("0.1"),
            side="short",
            mark_price=Decimal("50000"),
            funding_rate=Decimal("0.01"),
        )

        # +(0.1 * 50000 * 0.01) = +50
        assert funding == Decimal("50")

    def test_calculate_funding_long_negative_rate(self):
        """Longs receive when rate is negative."""
        calculator = FundingCalculator()

        funding = calculator.calculate_funding(
            position_size=Decimal("0.1"),
            side="long",
            mark_price=Decimal("50000"),
            funding_rate=Decimal("-0.01"),
        )

        # -(-50) = +50
        assert funding == Decimal("50")

    def test_calculate_funding_zero_position(self):
        """Zero position has no funding."""
        calculator = FundingCalculator()

        funding = calculator.calculate_funding(
            position_size=Decimal("0"),
            side="long",
            mark_price=Decimal("50000"),
            funding_rate=Decimal("0.01"),
        )

        assert funding == Decimal("0")

    def test_is_funding_due_checks_next_funding_time(self):
        """is_funding_due checks next_funding_time if provided."""
        calculator = FundingCalculator()

        past = datetime.now() - timedelta(hours=1)
        future = datetime.now() + timedelta(hours=1)

        assert calculator.is_funding_due(None, past) is True
        assert calculator.is_funding_due(None, future) is False

    def test_is_funding_due_checks_interval(self):
        """is_funding_due checks interval if no next_funding_time."""
        calculator = FundingCalculator()

        past_8h = datetime.now() - timedelta(hours=9)
        past_6h = datetime.now() - timedelta(hours=6)

        assert calculator.is_funding_due(past_8h, None) is True
        assert calculator.is_funding_due(past_6h, None) is False

    def test_is_funding_due_first_funding(self):
        """is_funding_due returns True for first funding."""
        calculator = FundingCalculator()

        assert calculator.is_funding_due(None, None) is True

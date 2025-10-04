"""
Unit tests for PositionValuer.

Tests position valuation logic with mark prices, staleness detection,
and PnL tracker integration.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.features.brokerages.core.interfaces import Position
from bot_v2.features.live_trade.pnl_tracker import PnLTracker
from bot_v2.features.live_trade.portfolio_valuation import MarkDataSource
from bot_v2.features.live_trade.position_valuer import PositionValuation, PositionValuer


class TestSinglePositionValuation:
    """Tests for valuing individual positions."""

    @pytest.fixture
    def mark_source(self):
        """Create mark source with 30s staleness threshold."""
        return MarkDataSource(staleness_threshold_seconds=30)

    @pytest.fixture
    def pnl_tracker(self):
        """Create fresh PnL tracker."""
        return PnLTracker()

    def test_zero_quantity_position_returns_none(self, mark_source, pnl_tracker):
        """Test zero quantity position is skipped."""
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("0"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )

        result = PositionValuer.value_position("BTC-USD", position, mark_source, pnl_tracker)

        assert result is None

    def test_missing_mark_price_returns_none(self, mark_source, pnl_tracker):
        """Test position without mark price returns None."""
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )

        # Don't update mark_source
        result = PositionValuer.value_position("BTC-USD", position, mark_source, pnl_tracker)

        assert result is None

    def test_valid_position_returns_valuation(self, mark_source, pnl_tracker):
        """Test valid position returns PositionValuation."""
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.5"),
            entry_price=Decimal("48000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )
        mark_source.update_mark("BTC-USD", Decimal("50000"))

        result = PositionValuer.value_position("BTC-USD", position, mark_source, pnl_tracker)

        assert result is not None
        assert isinstance(result, PositionValuation)
        assert result.symbol == "BTC-USD"

    def test_stale_mark_detected(self, pnl_tracker):
        """Test stale mark is detected and flagged."""
        mark_source = MarkDataSource(staleness_threshold_seconds=1)
        old_time = datetime.now() - timedelta(seconds=5)
        mark_source.update_mark("BTC-USD", Decimal("50000"), old_time)

        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )

        result = PositionValuer.value_position("BTC-USD", position, mark_source, pnl_tracker)

        assert result is not None
        assert result.is_stale is True

    def test_fresh_mark_not_stale(self, mark_source, pnl_tracker):
        """Test fresh mark is not flagged as stale."""
        mark_source.update_mark("BTC-USD", Decimal("50000"))

        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )

        result = PositionValuer.value_position("BTC-USD", position, mark_source, pnl_tracker)

        assert result is not None
        assert result.is_stale is False


class TestPositionSideDetection:
    """Tests for position side (long/short) detection."""

    @pytest.fixture
    def mark_source(self):
        """Create mark source."""
        source = MarkDataSource()
        source.update_mark("BTC-USD", Decimal("50000"))
        return source

    @pytest.fixture
    def pnl_tracker(self):
        """Create PnL tracker."""
        return PnLTracker()

    def test_long_position_side(self, mark_source, pnl_tracker):
        """Test positive quantity is detected as long."""
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )

        result = PositionValuer.value_position("BTC-USD", position, mark_source, pnl_tracker)

        assert result.side == "long"

    def test_short_position_side(self, mark_source, pnl_tracker):
        """Test negative quantity is detected as short."""
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("-1.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="short",
        )

        result = PositionValuer.value_position("BTC-USD", position, mark_source, pnl_tracker)

        assert result.side == "short"


class TestNotionalValueCalculation:
    """Tests for notional value calculation."""

    @pytest.fixture
    def mark_source(self):
        """Create mark source."""
        source = MarkDataSource()
        source.update_mark("BTC-USD", Decimal("50000"))
        source.update_mark("ETH-USD", Decimal("3000"))
        return source

    @pytest.fixture
    def pnl_tracker(self):
        """Create PnL tracker."""
        return PnLTracker()

    def test_notional_value_long_position(self, mark_source, pnl_tracker):
        """Test notional value for long position."""
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("2.5"),
            entry_price=Decimal("48000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )

        result = PositionValuer.value_position("BTC-USD", position, mark_source, pnl_tracker)

        expected_notional = Decimal("2.5") * Decimal("50000")
        assert result.notional_value == expected_notional

    def test_notional_value_short_position(self, mark_source, pnl_tracker):
        """Test notional value uses absolute quantity for short."""
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("-1.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="short",
        )

        result = PositionValuer.value_position("BTC-USD", position, mark_source, pnl_tracker)

        # Should use abs(quantity)
        expected_notional = Decimal("1.5") * Decimal("50000")
        assert result.notional_value == expected_notional

    def test_notional_value_small_quantity(self, mark_source, pnl_tracker):
        """Test notional value with small quantity."""
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.001"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )

        result = PositionValuer.value_position("BTC-USD", position, mark_source, pnl_tracker)

        expected_notional = Decimal("0.001") * Decimal("50000")
        assert result.notional_value == expected_notional

    def test_notional_value_different_asset(self, mark_source, pnl_tracker):
        """Test notional value calculation for different asset."""
        position = Position(
            symbol="ETH-USD",
            quantity=Decimal("10.0"),
            entry_price=Decimal("2800"),
            mark_price=Decimal("3000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )

        result = PositionValuer.value_position("ETH-USD", position, mark_source, pnl_tracker)

        expected_notional = Decimal("10.0") * Decimal("3000")
        assert result.notional_value == expected_notional


class TestPnLTrackerIntegration:
    """Tests for PnL tracker integration."""

    @pytest.fixture
    def mark_source(self):
        """Create mark source."""
        source = MarkDataSource()
        source.update_mark("BTC-USD", Decimal("50000"))
        return source

    @pytest.fixture
    def pnl_tracker(self):
        """Create PnL tracker with existing position."""
        tracker = PnLTracker()
        # Add a position with some PnL
        tracker.update_position("BTC-USD", "buy", Decimal("1.0"), Decimal("48000"))
        return tracker

    def test_pnl_tracker_mark_update(self, mark_source, pnl_tracker):
        """Test mark price updates PnL tracker without error."""
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("48000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )

        # Should update PnL tracker's mark without error
        result = PositionValuer.value_position("BTC-USD", position, mark_source, pnl_tracker)

        # Verify position was valued (mark update succeeded)
        assert result is not None
        assert result.mark_price == Decimal("50000")

    def test_unrealized_pnl_from_tracker(self, mark_source, pnl_tracker):
        """Test unrealized PnL comes from tracker."""
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("48000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )

        result = PositionValuer.value_position("BTC-USD", position, mark_source, pnl_tracker)

        # Should have unrealized PnL from tracker (50000 - 48000) * 1.0 = 2000
        assert result.unrealized_pnl == Decimal("2000")

    def test_avg_entry_price_from_tracker(self, mark_source, pnl_tracker):
        """Test avg entry price comes from tracker."""
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("48000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            side="long",
        )

        result = PositionValuer.value_position("BTC-USD", position, mark_source, pnl_tracker)

        assert result.avg_entry_price == Decimal("48000")


class TestMultiplePositionsValuation:
    """Tests for value_positions batch operation."""

    @pytest.fixture
    def mark_source(self):
        """Create mark source with multiple marks."""
        source = MarkDataSource()
        source.update_mark("BTC-USD", Decimal("50000"))
        source.update_mark("ETH-USD", Decimal("3000"))
        return source

    @pytest.fixture
    def pnl_tracker(self):
        """Create PnL tracker."""
        return PnLTracker()

    def test_value_multiple_positions(self, mark_source, pnl_tracker):
        """Test valuing multiple positions."""
        positions = {
            "BTC-USD": Position(
                symbol="BTC-USD",
                quantity=Decimal("1.0"),
                entry_price=Decimal("48000"),
                mark_price=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            ),
            "ETH-USD": Position(
                symbol="ETH-USD",
                quantity=Decimal("10.0"),
                entry_price=Decimal("2800"),
                mark_price=Decimal("3000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            ),
        }

        details, total_value, stale, missing = PositionValuer.value_positions(
            positions, mark_source, pnl_tracker
        )

        assert len(details) == 2
        assert "BTC-USD" in details
        assert "ETH-USD" in details
        assert total_value == Decimal("50000") + Decimal("30000")  # BTC + ETH
        assert len(stale) == 0
        assert len(missing) == 0

    def test_value_positions_with_missing_mark(self, mark_source, pnl_tracker):
        """Test missing mark is tracked."""
        positions = {
            "BTC-USD": Position(
                symbol="BTC-USD",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000"),
                mark_price=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            ),
            "MISSING-USD": Position(
                symbol="MISSING-USD",
                quantity=Decimal("1.0"),
                entry_price=Decimal("100"),
                mark_price=Decimal("100"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            ),
        }

        details, total_value, stale, missing = PositionValuer.value_positions(
            positions, mark_source, pnl_tracker
        )

        assert len(details) == 1  # Only BTC-USD
        assert "BTC-USD" in details
        assert "MISSING-USD" not in details
        assert "MISSING-USD" in missing

    def test_value_positions_with_stale_mark(self, pnl_tracker):
        """Test stale mark is tracked."""
        mark_source = MarkDataSource(staleness_threshold_seconds=1)
        old_time = datetime.now() - timedelta(seconds=5)
        mark_source.update_mark("BTC-USD", Decimal("50000"), old_time)
        mark_source.update_mark("ETH-USD", Decimal("3000"))

        positions = {
            "BTC-USD": Position(
                symbol="BTC-USD",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000"),
                mark_price=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            ),
            "ETH-USD": Position(
                symbol="ETH-USD",
                quantity=Decimal("1.0"),
                entry_price=Decimal("3000"),
                mark_price=Decimal("3000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            ),
        }

        details, total_value, stale, missing = PositionValuer.value_positions(
            positions, mark_source, pnl_tracker
        )

        assert len(details) == 2
        assert "BTC-USD" in stale
        assert "ETH-USD" not in stale

    def test_value_positions_skips_zero_quantity(self, mark_source, pnl_tracker):
        """Test zero quantity positions are skipped."""
        positions = {
            "BTC-USD": Position(
                symbol="BTC-USD",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000"),
                mark_price=Decimal("50000"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            ),
            "ZERO-USD": Position(
                symbol="ZERO-USD",
                quantity=Decimal("0"),
                entry_price=Decimal("100"),
                mark_price=Decimal("100"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
            ),
        }

        details, total_value, stale, missing = PositionValuer.value_positions(
            positions, mark_source, pnl_tracker
        )

        assert len(details) == 1
        assert "BTC-USD" in details
        assert "ZERO-USD" not in details
        assert "ZERO-USD" not in missing  # Zero quantity, not missing mark


class TestPositionValuationToDict:
    """Tests for PositionValuation.to_dict()."""

    def test_to_dict_serialization(self):
        """Test PositionValuation serializes to dict."""
        valuation = PositionValuation(
            symbol="BTC-USD",
            side="long",
            quantity=Decimal("1.5"),
            mark_price=Decimal("50000"),
            notional_value=Decimal("75000"),
            unrealized_pnl=Decimal("3000"),
            realized_pnl=Decimal("500"),
            funding_paid=Decimal("-50"),
            avg_entry_price=Decimal("48000"),
            is_stale=False,
        )

        result = valuation.to_dict()

        assert result["side"] == "long"
        assert result["quantity"] == Decimal("1.5")
        assert result["mark_price"] == Decimal("50000")
        assert result["notional_value"] == Decimal("75000")
        assert result["unrealized_pnl"] == Decimal("3000")
        assert result["realized_pnl"] == Decimal("500")
        assert result["funding_paid"] == Decimal("-50")
        assert result["avg_entry_price"] == Decimal("48000")
        assert result["is_stale"] is False

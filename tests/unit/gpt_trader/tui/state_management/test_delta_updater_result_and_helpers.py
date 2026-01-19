"""Tests for DeltaResult and StateDeltaUpdater helper methods."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.tui.state_management.delta_updater import DeltaResult, StateDeltaUpdater


class TestDeltaResult:
    """Test DeltaResult dataclass."""

    def test_initial_state_no_changes(self):
        """Test new DeltaResult has no changes."""
        result = DeltaResult()

        assert result.has_changes is False
        assert result.changed_fields == []
        assert result.details == {}

    def test_add_change_sets_has_changes(self):
        """Test adding a change sets has_changes flag."""
        result = DeltaResult()
        result.add_change("field", "old", "new")

        assert result.has_changes is True
        assert "field" in result.changed_fields
        assert result.details["field"] == ("old", "new")

    def test_add_multiple_changes(self):
        """Test adding multiple changes."""
        result = DeltaResult()
        result.add_change("field1", "old1", "new1")
        result.add_change("field2", "old2", "new2")

        assert len(result.changed_fields) == 2
        assert len(result.details) == 2


class TestStateDeltaUpdaterHelpers:
    """Test StateDeltaUpdater helper methods."""

    def test_should_update_component_no_changes(self):
        """Test should_update returns False when no changes."""
        updater = StateDeltaUpdater()
        delta = DeltaResult()

        assert not updater.should_update_component("market", delta)

    def test_should_update_component_with_changes(self):
        """Test should_update returns True when there are changes."""
        updater = StateDeltaUpdater()
        delta = DeltaResult()
        delta.add_change("prices.BTC-USD", "50000", "51000")

        assert updater.should_update_component("market", delta)

    def test_float_equal_within_epsilon(self):
        """Test float comparison with epsilon tolerance."""
        updater = StateDeltaUpdater()

        assert updater._float_equal(1.0, 1.0)
        assert updater._float_equal(1.0, 1.0 + 1e-10)  # Within epsilon
        assert not updater._float_equal(1.0, 1.1)  # Beyond epsilon

    def test_decimal_equal_within_epsilon(self):
        """Test decimal comparison with epsilon tolerance."""
        updater = StateDeltaUpdater()

        assert updater._decimal_equal(Decimal("1.0"), Decimal("1.0"))
        assert updater._decimal_equal(Decimal("1.0"), Decimal("1.000000001"))
        assert not updater._decimal_equal(Decimal("1.0"), Decimal("1.1"))

    def test_decimal_equal_handles_none(self):
        """Test decimal comparison handles None values."""
        updater = StateDeltaUpdater()

        assert updater._decimal_equal(None, None)
        assert not updater._decimal_equal(Decimal("1.0"), None)
        assert not updater._decimal_equal(None, Decimal("1.0"))

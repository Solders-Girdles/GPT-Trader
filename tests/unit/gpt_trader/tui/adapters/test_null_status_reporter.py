"""
Tests for NullStatusReporter adapter.

Verifies the null adapter implements the StatusReporter interface correctly
and returns appropriate unavailable status data.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.adapters.null_status_reporter import (
    NullStatusReporter,
    _create_unavailable_status,
)


class TestNullStatusReporter:
    """Test suite for NullStatusReporter."""

    def test_is_null_reporter_property(self) -> None:
        """NullStatusReporter should have is_null_reporter=True."""
        reporter = NullStatusReporter()
        assert reporter.is_null_reporter is True

    def test_add_observer_stores_callback(self) -> None:
        """add_observer should store the callback without error."""
        reporter = NullStatusReporter()
        callback = MagicMock()

        reporter.add_observer(callback)

        assert callback in reporter._observers

    def test_add_observer_prevents_duplicates(self) -> None:
        """add_observer should not add the same callback twice."""
        reporter = NullStatusReporter()
        callback = MagicMock()

        reporter.add_observer(callback)
        reporter.add_observer(callback)

        assert reporter._observers.count(callback) == 1

    def test_remove_observer_removes_callback(self) -> None:
        """remove_observer should remove the callback."""
        reporter = NullStatusReporter()
        callback = MagicMock()

        reporter.add_observer(callback)
        reporter.remove_observer(callback)

        assert callback not in reporter._observers

    def test_remove_observer_handles_missing_callback(self) -> None:
        """remove_observer should not error when callback not found."""
        reporter = NullStatusReporter()
        callback = MagicMock()

        # Should not raise
        reporter.remove_observer(callback)

    def test_get_status_returns_unavailable_status(self) -> None:
        """get_status should return a BotStatus indicating unavailable."""
        reporter = NullStatusReporter()
        status = reporter.get_status()

        assert status.healthy is False
        assert "StatusReporter not available" in status.health_issues[0]
        assert status.bot_id == "unavailable"
        assert status.version == "--"

    def test_get_status_has_empty_data(self) -> None:
        """get_status should return empty data for all components."""
        reporter = NullStatusReporter()
        status = reporter.get_status()

        # Engine should show not running
        assert status.engine.running is False
        assert status.engine.cycle_count == 0

        # Market should be empty
        assert status.market.symbols == []
        assert status.market.last_prices == {}

        # Positions should be empty
        assert status.positions.count == 0
        assert status.positions.positions == {}

        # Orders and trades should be empty
        assert status.orders == []
        assert status.trades == []

        # System should show unavailable
        assert status.system.connection_status == "UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_start_returns_none(self) -> None:
        """start should return None (no background task)."""
        reporter = NullStatusReporter()
        result = await reporter.start()

        assert result is None
        assert reporter._running is True

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self) -> None:
        """stop should set _running to False."""
        reporter = NullStatusReporter()
        await reporter.start()
        await reporter.stop()

        assert reporter._running is False


class TestCreateUnavailableStatus:
    """Test suite for _create_unavailable_status factory."""

    def test_creates_valid_bot_status(self) -> None:
        """Should create a valid BotStatus object."""
        status = _create_unavailable_status()

        # Basic attributes
        assert status.bot_id == "unavailable"
        assert status.healthy is False

        # timestamp should be set
        assert status.timestamp > 0

    def test_has_health_issues(self) -> None:
        """Should have appropriate health issues listed."""
        status = _create_unavailable_status()

        assert len(status.health_issues) == 1
        assert "StatusReporter not available" in status.health_issues[0]

    def test_all_components_initialized(self) -> None:
        """All status components should be initialized."""
        status = _create_unavailable_status()

        # All components should exist
        assert status.engine is not None
        assert status.market is not None
        assert status.positions is not None
        assert status.account is not None
        assert status.strategy is not None
        assert status.risk is not None
        assert status.system is not None
        assert status.heartbeat is not None

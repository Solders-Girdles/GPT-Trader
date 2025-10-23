"""Tests for the ReduceOnlyModeStateManager."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest

from bot_v2.orchestration.state_manager import (
    ReduceOnlyModeSource,
    ReduceOnlyModeState,
    ReduceOnlyModeStateManager,
    StateChangeRequest,
    create_reduce_only_state_manager,
)


class TestReduceOnlyModeSource:
    """Test the ReduceOnlyModeSource enum."""

    def test_source_values(self) -> None:
        """Test that all expected source values are present."""
        expected_sources = {
            "config",
            "risk_manager",
            "runtime_coordinator",
            "guard_failure",
            "startup_reconcile_failed",
            "derivatives_not_accessible",
            "daily_loss_limit",
            "user_request",
            "system_monitor",
        }
        actual_sources = {source.value for source in ReduceOnlyModeSource}
        assert actual_sources == expected_sources


class TestReduceOnlyModeState:
    """Test the ReduceOnlyModeState dataclass."""

    def test_default_state(self) -> None:
        """Test default state values."""
        state = ReduceOnlyModeState()
        assert state.enabled is False
        assert state.reason is None
        assert state.source is None
        assert state.timestamp is None
        assert state.previous_state is False

    def test_state_initialization(self) -> None:
        """Test state initialization with values."""
        timestamp = datetime.datetime.utcnow()
        state = ReduceOnlyModeState(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
            timestamp=timestamp,
            previous_state=False,
        )
        assert state.enabled is True
        assert state.reason == "test"
        assert state.source == ReduceOnlyModeSource.CONFIG
        assert state.timestamp == timestamp
        assert state.previous_state is False

    def test_to_dict(self) -> None:
        """Test state serialization to dictionary."""
        timestamp = datetime.datetime.utcnow()
        state = ReduceOnlyModeState(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
            timestamp=timestamp,
            previous_state=False,
        )
        expected = {
            "enabled": True,
            "reason": "test",
            "source": "config",
            "timestamp": timestamp.isoformat(),
            "previous_state": False,
        }
        assert state.to_dict() == expected


class TestStateChangeRequest:
    """Test the StateChangeRequest dataclass."""

    def test_default_request(self) -> None:
        """Test default request values."""
        request = StateChangeRequest(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
        )
        assert request.enabled is True
        assert request.reason == "test"
        assert request.source == ReduceOnlyModeSource.CONFIG
        assert request.metadata == {}
        assert isinstance(request.timestamp, datetime.datetime)

    def test_request_with_metadata(self) -> None:
        """Test request with metadata."""
        metadata = {"key": "value"}
        request = StateChangeRequest(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
            metadata=metadata,
        )
        assert request.metadata == metadata


class TestReduceOnlyModeStateManager:
    """Test the ReduceOnlyModeStateManager class."""

    def test_initialization(self) -> None:
        """Test manager initialization."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store)

        assert manager.event_store == event_store
        assert manager.is_reduce_only_mode is False
        assert manager._validation_enabled is True
        assert len(manager._audit_log) == 0
        assert len(manager._state_listeners) == 0

    def test_initialization_with_custom_values(self) -> None:
        """Test manager initialization with custom values."""
        event_store = MagicMock()

        def now_provider() -> datetime.datetime:
            return datetime.datetime(2023, 1, 1)

        manager = ReduceOnlyModeStateManager(
            event_store=event_store,
            now_provider=now_provider,
            initial_state=True,
            validation_enabled=False,
        )

        assert manager.is_reduce_only_mode is True
        assert manager._validation_enabled is False

    def test_set_reduce_only_mode_success(self) -> None:
        """Test successful state change."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store)

        # Enable reduce-only mode
        changed = manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
        )

        assert changed is True
        assert manager.is_reduce_only_mode is True
        assert manager._state.enabled is True
        assert manager._state.reason == "test"
        assert manager._state.source == ReduceOnlyModeSource.CONFIG
        assert manager._state.previous_state is False
        assert len(manager._audit_log) == 1

    def test_set_reduce_only_mode_no_change(self) -> None:
        """Test setting state to same value."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store)

        # Set to same value
        changed = manager.set_reduce_only_mode(
            enabled=False,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
        )

        assert changed is False
        assert len(manager._audit_log) == 0

    def test_set_reduce_only_mode_with_metadata(self) -> None:
        """Test state change with metadata."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store)

        metadata = {"key": "value"}
        changed = manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.DAILY_LOSS_LIMIT,
            metadata=metadata,
        )

        assert changed is True
        assert manager._audit_log[0].metadata == metadata

    def test_validation_empty_reason(self) -> None:
        """Test validation with empty reason."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store, validation_enabled=True)

        with pytest.raises(ValueError, match="Reason cannot be empty"):
            manager.set_reduce_only_mode(
                enabled=True,
                reason="",
                source=ReduceOnlyModeSource.CONFIG,
            )

    def test_validation_disabled(self) -> None:
        """Test that validation can be disabled."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store, validation_enabled=False)

        # Should not raise error even with empty reason
        changed = manager.set_reduce_only_mode(
            enabled=True,
            reason="",
            source=ReduceOnlyModeSource.CONFIG,
        )

        assert changed is True

    def test_validation_daily_loss_limit_metadata(self) -> None:
        """Test validation for daily loss limit requires metadata."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store, validation_enabled=True)

        with pytest.raises(ValueError, match="loss_amount"):
            manager.set_reduce_only_mode(
                enabled=True,
                reason="test",
                source=ReduceOnlyModeSource.DAILY_LOSS_LIMIT,
            )

    def test_add_remove_state_listener(self) -> None:
        """Test adding and removing state listeners."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store)

        listener = MagicMock()

        # Add listener
        manager.add_state_listener(listener)
        assert len(manager._state_listeners) == 1

        # Change state to trigger listener
        manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
        )

        listener.assert_called_once()

        # Remove listener
        manager.remove_state_listener(listener)
        assert len(manager._state_listeners) == 0

    def test_state_listener_error_handling(self) -> None:
        """Test that listener errors don't break state changes."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store)

        # Add a listener that raises an exception
        def bad_listener(state: ReduceOnlyModeState) -> None:
            raise Exception("Listener error")

        manager.add_state_listener(bad_listener)

        # State change should still succeed
        changed = manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
        )

        assert changed is True
        assert manager.is_reduce_only_mode is True

    def test_get_audit_log(self) -> None:
        """Test retrieving audit log."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store)

        # Make some changes
        manager.set_reduce_only_mode(
            enabled=True,
            reason="test1",
            source=ReduceOnlyModeSource.CONFIG,
        )
        manager.set_reduce_only_mode(
            enabled=False,
            reason="test2",
            source=ReduceOnlyModeSource.RISK_MANAGER,
        )

        audit_log = manager.get_audit_log()
        assert len(audit_log) == 2
        assert audit_log[0].reason == "test2"  # Most recent first
        assert audit_log[1].reason == "test1"

    def test_get_audit_log_with_limit(self) -> None:
        """Test retrieving audit log with limit."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store)

        # Make some changes
        for i in range(5):
            manager.set_reduce_only_mode(
                enabled=i % 2 == 0,
                reason=f"test{i}",
                source=ReduceOnlyModeSource.CONFIG,
            )

        audit_log = manager.get_audit_log(limit=3)
        assert len(audit_log) == 3
        assert audit_log[0].reason == "test4"  # Most recent first

    def test_clear_audit_log(self) -> None:
        """Test clearing audit log."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store)

        # Make some changes
        manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
        )

        assert len(manager._audit_log) == 1

        manager.clear_audit_log()
        assert len(manager._audit_log) == 0

    def test_reset_state(self) -> None:
        """Test resetting state."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store)

        # Change state
        manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
        )

        # Reset
        changed = manager.reset_state(enabled=False, reason="reset")

        assert changed is True
        assert manager.is_reduce_only_mode is False
        assert manager._state.reason == "reset"
        assert manager._state.source == ReduceOnlyModeSource.SYSTEM_MONITOR

    def test_get_state_summary(self) -> None:
        """Test getting state summary."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store)

        # Make a change
        manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
        )

        summary = manager.get_state_summary()

        assert summary["current_state"]["enabled"] is True
        assert summary["change_count"] == 1
        assert summary["listener_count"] == 0
        assert summary["validation_enabled"] is True
        assert "last_change" in summary

    @patch("bot_v2.orchestration.state_manager.emit_metric")
    def test_emit_state_change_metric(self, mock_emit_metric: MagicMock) -> None:
        """Test that metrics are emitted for state changes."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store)

        manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
        )

        mock_emit_metric.assert_called_once()

    @patch("bot_v2.orchestration.state_manager.emit_metric")
    def test_emit_metric_error_handling(self, mock_emit_metric: MagicMock) -> None:
        """Test that metric emission errors don't break state changes."""
        event_store = MagicMock()
        manager = ReduceOnlyModeStateManager(event_store)

        mock_emit_metric.side_effect = Exception("Metric error")

        # State change should still succeed
        changed = manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
        )

        assert changed is True
        assert manager.is_reduce_only_mode is True


class TestCreateReduceOnlyStateManager:
    """Test the create_reduce_only_state_manager factory function."""

    def test_factory_function(self) -> None:
        """Test that factory function creates manager correctly."""
        event_store = MagicMock()

        manager = create_reduce_only_state_manager(
            event_store=event_store,
            initial_state=True,
            validation_enabled=False,
        )

        assert isinstance(manager, ReduceOnlyModeStateManager)
        assert manager.event_store == event_store
        assert manager.is_reduce_only_mode is True
        assert manager._validation_enabled is False

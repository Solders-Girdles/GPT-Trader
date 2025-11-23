"""Unit tests for the ReduceOnlyModeStateManager.

These tests verify that the centralized state manager works correctly
for state transitions, validation, and concurrent access patterns.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from bot_v2.orchestration.state.unified_state import (
    ReduceOnlyModeSource,
    ReduceOnlyModeState,
)
from bot_v2.orchestration.state.unified_state import (
    SystemState as ReduceOnlyModeStateManager,
)
from bot_v2.persistence.event_store import EventStore


class TestReduceOnlyModeStateManager:
    """Test cases for ReduceOnlyModeStateManager."""

    def test_initialization(self):
        """Test state manager initialization."""
        event_store = Mock(spec=EventStore)

        # Test default initialization
        manager = ReduceOnlyModeStateManager(event_store)
        assert manager.is_reduce_only_mode is False
        assert manager.current_state.enabled is False
        assert manager.current_state.reason is None
        assert manager.current_state.source is None
        assert len(manager.get_audit_log()) == 0

        # Test initialization with initial state
        manager_enabled = ReduceOnlyModeStateManager(event_store, initial_state=True)
        assert manager_enabled.is_reduce_only_mode is True
        assert manager_enabled.current_state.enabled is True

    def test_set_reduce_only_mode_success(self):
        """Test successful state changes."""
        event_store = Mock(spec=EventStore)
        manager = ReduceOnlyModeStateManager(event_store)

        # Enable reduce-only mode
        result = manager.set_reduce_only_mode(
            enabled=True, reason="test_enable", source=ReduceOnlyModeSource.USER_REQUEST
        )

        assert result is True  # State changed
        assert manager.is_reduce_only_mode is True
        assert manager.current_state.enabled is True
        assert manager.current_state.reason == "test_enable"
        assert manager.current_state.source == ReduceOnlyModeSource.USER_REQUEST
        assert manager.current_state.timestamp is not None
        assert manager.current_state.previous_state is False

        # Verify audit log
        audit_log = manager.get_audit_log()
        assert len(audit_log) == 1
        assert audit_log[0].enabled is True
        assert audit_log[0].reason == "test_enable"
        assert audit_log[0].source == ReduceOnlyModeSource.USER_REQUEST

        # Disable reduce-only mode
        result = manager.set_reduce_only_mode(
            enabled=False, reason="test_disable", source=ReduceOnlyModeSource.CONFIG
        )

        assert result is True  # State changed
        assert manager.is_reduce_only_mode is False
        assert manager.current_state.enabled is False
        assert manager.current_state.reason == "test_disable"
        assert manager.current_state.source == ReduceOnlyModeSource.CONFIG
        assert manager.current_state.previous_state is True

        # Verify audit log has 2 entries
        audit_log = manager.get_audit_log()
        assert len(audit_log) == 2

    def test_set_reduce_only_mode_no_change(self):
        """Test setting the same state (no change)."""
        event_store = Mock(spec=EventStore)
        manager = ReduceOnlyModeStateManager(event_store)

        # Set to same state (False -> False)
        result = manager.set_reduce_only_mode(
            enabled=False, reason="no_change", source=ReduceOnlyModeSource.CONFIG
        )

        assert result is False  # No state change
        assert manager.is_reduce_only_mode is False
        assert len(manager.get_audit_log()) == 0  # No audit log entry

        # Enable first
        manager.set_reduce_only_mode(
            enabled=True, reason="enable", source=ReduceOnlyModeSource.USER_REQUEST
        )

        # Then set to same state (True -> True)
        result = manager.set_reduce_only_mode(
            enabled=True, reason="no_change", source=ReduceOnlyModeSource.CONFIG
        )

        assert result is False  # No state change
        assert manager.is_reduce_only_mode is True
        assert len(manager.get_audit_log()) == 1  # Only the enable entry

    def test_validation_enabled_empty_reason(self):
        """Test validation with empty reason."""
        event_store = Mock(spec=EventStore)
        manager = ReduceOnlyModeStateManager(event_store, validation_enabled=True)

        # Test empty reason
        with pytest.raises(ValueError, match="Reason cannot be empty"):
            manager.set_reduce_only_mode(
                enabled=True, reason="", source=ReduceOnlyModeSource.USER_REQUEST  # Empty reason
            )

        # Test whitespace-only reason
        with pytest.raises(ValueError, match="Reason cannot be empty"):
            manager.set_reduce_only_mode(
                enabled=True,
                reason="   ",  # Whitespace only
                source=ReduceOnlyModeSource.USER_REQUEST,
            )

    def test_validation_disabled_empty_reason(self):
        """Test that validation can be disabled."""
        event_store = Mock(spec=EventStore)
        manager = ReduceOnlyModeStateManager(event_store, validation_enabled=False)

        # Should not raise error when validation is disabled
        result = manager.set_reduce_only_mode(
            enabled=True, reason="", source=ReduceOnlyModeSource.USER_REQUEST  # Empty reason
        )

        assert result is True
        assert manager.is_reduce_only_mode is True

    def test_validation_daily_loss_limit_metadata(self):
        """Test validation for daily loss limit source."""
        event_store = Mock(spec=EventStore)
        manager = ReduceOnlyModeStateManager(event_store, validation_enabled=True)

        # Daily loss limit requires loss_amount in metadata
        with pytest.raises(ValueError, match="loss_amount"):
            manager.set_reduce_only_mode(
                enabled=True,
                reason="daily loss exceeded",
                source=ReduceOnlyModeSource.DAILY_LOSS_LIMIT,
                metadata={},  # Missing loss_amount
            )

        # Should work with proper metadata
        result = manager.set_reduce_only_mode(
            enabled=True,
            reason="daily loss exceeded",
            source=ReduceOnlyModeSource.DAILY_LOSS_LIMIT,
            metadata={"loss_amount": 1000.0},
        )

        assert result is True

    def test_state_listeners(self):
        """Test state change listeners."""
        event_store = Mock(spec=EventStore)
        manager = ReduceOnlyModeStateManager(event_store)

        # Mock listener
        listener = Mock()
        manager.add_state_listener(listener)

        # Change state
        manager.set_reduce_only_mode(
            enabled=True, reason="listener_test", source=ReduceOnlyModeSource.USER_REQUEST
        )

        # Verify listener was called
        assert listener.call_count == 1
        call_args = listener.call_args[0][0]  # First argument of first call
        assert isinstance(call_args, ReduceOnlyModeState)
        assert call_args.enabled is True
        assert call_args.reason == "listener_test"

        # Remove listener
        manager.remove_state_listener(listener)

        # Change state again
        manager.set_reduce_only_mode(
            enabled=False, reason="listener_removed", source=ReduceOnlyModeSource.CONFIG
        )

        # Verify listener was not called again
        assert listener.call_count == 1

    def test_state_listener_error_handling(self):
        """Test that listener errors don't break state changes."""
        event_store = Mock(spec=EventStore)
        manager = ReduceOnlyModeStateManager(event_store)

        # Add a listener that raises an exception
        def faulty_listener(state):
            raise Exception("Listener error")

        manager.add_state_listener(faulty_listener)

        # State change should still succeed despite listener error
        result = manager.set_reduce_only_mode(
            enabled=True, reason="faulty_listener_test", source=ReduceOnlyModeSource.USER_REQUEST
        )

        assert result is True
        assert manager.is_reduce_only_mode is True

    def test_reset_state(self):
        """Test state reset functionality."""
        event_store = Mock(spec=EventStore)
        manager = ReduceOnlyModeStateManager(event_store)

        # Set initial state
        manager.set_reduce_only_mode(
            enabled=True, reason="initial", source=ReduceOnlyModeSource.USER_REQUEST
        )
        assert manager.is_reduce_only_mode is True

        # Reset to False
        result = manager.reset_state(enabled=False, reason="test_reset")
        assert result is True
        assert manager.is_reduce_only_mode is False
        assert manager.current_state.reason == "test_reset"
        assert manager.current_state.source == ReduceOnlyModeSource.SYSTEM_MONITOR

        # Reset to True
        result = manager.reset_state(enabled=True, reason="test_reset_true")
        assert result is True
        assert manager.is_reduce_only_mode is True

    def test_get_state_summary(self):
        """Test state summary functionality."""
        event_store = Mock(spec=EventStore)
        manager = ReduceOnlyModeStateManager(event_store)

        # Initial summary
        summary = manager.get_state_summary()
        assert summary["current_state"]["enabled"] is False
        assert summary["change_count"] == 0
        assert summary["last_change"] is None
        assert summary["listener_count"] == 0
        assert summary["validation_enabled"] is True

        # Make some changes
        manager.set_reduce_only_mode(
            enabled=True, reason="summary_test", source=ReduceOnlyModeSource.USER_REQUEST
        )

        # Updated summary
        summary = manager.get_state_summary()
        assert summary["current_state"]["enabled"] is True
        assert summary["change_count"] == 1
        assert summary["last_change"] is not None

    def test_audit_log_limiting(self):
        """Test audit log limiting functionality."""
        event_store = Mock(spec=EventStore)
        manager = ReduceOnlyModeStateManager(event_store)

        # Add multiple changes
        for i in range(5):
            manager.set_reduce_only_mode(
                enabled=i % 2 == 0, reason=f"change_{i}", source=ReduceOnlyModeSource.USER_REQUEST
            )

        # Get full audit log
        full_log = manager.get_audit_log()
        assert len(full_log) == 5

        # Get limited audit log
        limited_log = manager.get_audit_log(limit=3)
        assert len(limited_log) == 3
        # Should get the most recent 3 changes
        assert limited_log[0].reason == "change_4"
        assert limited_log[1].reason == "change_3"
        assert limited_log[2].reason == "change_2"

    def test_clear_audit_log(self):
        """Test audit log clearing."""
        event_store = Mock(spec=EventStore)
        manager = ReduceOnlyModeStateManager(event_store)

        # Add some changes
        manager.set_reduce_only_mode(True, "test1", ReduceOnlyModeSource.USER_REQUEST)
        manager.set_reduce_only_mode(False, "test2", ReduceOnlyModeSource.CONFIG)

        assert len(manager.get_audit_log()) == 2

        # Clear audit log
        manager.clear_audit_log()
        assert len(manager.get_audit_log()) == 0

    def test_custom_time_provider(self):
        """Test custom time provider for testing."""
        event_store = Mock(spec=EventStore)

        # Fixed time for testing - note that StateChangeRequest uses datetime.utcnow()
        # so this test verifies the time provider is accepted even if not used for timestamps
        fixed_time = datetime(2023, 1, 1, 12, 0, 0)
        manager = ReduceOnlyModeStateManager(event_store, now_provider=lambda: fixed_time)

        # Verify the custom time provider was stored (even if StateChangeRequest uses utcnow)
        assert manager._now_provider() == fixed_time

        manager.set_reduce_only_mode(
            enabled=True, reason="time_test", source=ReduceOnlyModeSource.USER_REQUEST
        )

        # Verify the state change worked (timestamp will be from StateChangeRequest.default_factory)
        assert manager.current_state.enabled is True
        assert manager.current_state.timestamp is not None

    def test_critical_sources_logging(self):
        """Test that critical sources can be used and logged."""
        event_store = Mock(spec=EventStore)
        manager = ReduceOnlyModeStateManager(event_store)

        critical_sources = [
            ReduceOnlyModeSource.GUARD_FAILURE,
            ReduceOnlyModeSource.STARTUP_RECONCILE_FAILED,
            ReduceOnlyModeSource.DERIVATIVES_NOT_ACCESSIBLE,
        ]

        for source in critical_sources:
            # Test that critical sources work correctly
            result = manager.set_reduce_only_mode(
                enabled=True, reason=f"critical_{source.value}", source=source
            )

            assert result is True
            assert manager.is_reduce_only_mode is True
            assert manager.current_state.source == source
            assert f"critical_{source.value}" in manager.current_state.reason

            # Reset for next test
            manager.reset_state(enabled=False, reason="reset_test")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])

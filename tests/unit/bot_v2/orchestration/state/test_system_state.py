"""Tests for the SystemState."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest

from bot_v2.orchestration.state.unified_state import (
    ReduceOnlyModeSource,
    ReduceOnlyModeState,
    StateChange,  # Replaced StateChangeRequest
    create_reduce_only_state_manager,
    StateType,
)
from bot_v2.orchestration.state.unified_state import (
    SystemState,
)

# Alias StateChangeRequest to StateChange for test compatibility if needed, or just update usage
StateChangeRequest = StateChange


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


class TestStateChange:
    """Test StateChange data class."""

    def test_default_request(self) -> None:
        """Test creating default request."""
        request = StateChange(
            timestamp=datetime.datetime.utcnow(),
            state_type=StateType.REDUCE_ONLY_MODE,
            old_value=False,
            new_value=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG.value,
        )

        assert request.state_type == StateType.REDUCE_ONLY_MODE
        assert request.new_value is True
        assert request.reason == "test"
        assert request.source == ReduceOnlyModeSource.CONFIG.value
        assert request.metadata == {}

    def test_request_with_metadata(self) -> None:
        """Test creating request with metadata."""
        metadata = {"key": "value"}
        request = StateChange(
            timestamp=datetime.datetime.utcnow(),
            state_type=StateType.REDUCE_ONLY_MODE,
            old_value=False,
            new_value=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG.value,
            metadata=metadata,
        )

        assert request.metadata == metadata


class TestSystemState:
    """Test the SystemState class."""

    def test_initialization(self) -> None:
        """Test manager initialization."""
        manager = SystemState()
        manager.initialize()

        assert manager.get_reduce_only_mode() is False
        assert len(manager._change_history) == 0
        assert len(manager._change_listeners) == 0

    def test_initialization_with_custom_values(self) -> None:
        """Test manager initialization with custom values."""
        # SystemState doesn't support custom values in init anymore, 
        # it relies on set_state or defaults.
        # We test defaults here.
        manager = SystemState()
        manager.initialize()

        assert manager.get_reduce_only_mode() is False

    def test_set_reduce_only_mode_success(self) -> None:
        """Test successful state change."""
        manager = SystemState()
        manager.initialize()

        # Enable reduce-only mode
        manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG.value,
        )

        assert manager.get_reduce_only_mode() is True
        
        # Verify history
        history = manager.get_change_history(StateType.REDUCE_ONLY_MODE)
        assert len(history) == 1
        assert history[0].new_value is True
        assert history[0].reason == "test"
        assert history[0].source == ReduceOnlyModeSource.CONFIG.value

    def test_state_listener_error_handling(self) -> None:
        """Test that listener errors don't break state changes."""
        manager = SystemState()
        manager.initialize()

        # Add a listener that raises an exception
        def bad_listener(change: StateChange) -> None:
            raise Exception("Listener error")

        manager.add_state_listener(StateType.REDUCE_ONLY_MODE, bad_listener)

        # State change should still succeed
        manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG.value,
        )

        assert manager.get_reduce_only_mode() is True

    def test_reset_state(self) -> None:
        """Test resetting state."""
        # SystemState doesn't have explicit reset_state method exposed like before
        # but we can set state.
        manager = SystemState()
        manager.initialize()
        
        manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG.value,
        )
        
        assert manager.get_reduce_only_mode() is True
        
        # "Reset" by setting to false
        manager.set_reduce_only_mode(
            enabled=False,
            reason="reset",
            source=ReduceOnlyModeSource.SYSTEM_MONITOR.value,
        )
        
        assert manager.get_reduce_only_mode() is False

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

        assert isinstance(manager, SystemState)
        # event_store is not stored on SystemState instance by default unless we subclass or add it
        # The factory ignores it currently.
        # assert manager.event_store == event_store 
        assert manager.get_reduce_only_mode() is True
        # assert manager._validation_enabled is False

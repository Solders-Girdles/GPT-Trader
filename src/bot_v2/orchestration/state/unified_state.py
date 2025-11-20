"""Unified state management system for orchestration.

This module consolidates overlapping state management responsibilities from
multiple managers into a single, consistent system. It provides:

- Centralized state tracking for all orchestration state
- Consistent validation and audit logging
- Thread-safe operations
- Change notification system
- State history and rollback capabilities
- Integration with existing manager patterns
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any, cast

from .base import BaseManager


class ReduceOnlyModeSource(Enum):
    """Enumeration of valid sources for reduce_only_mode changes."""

    CONFIG = "config"
    RISK_MANAGER = "risk_manager"
    RUNTIME_COORDINATOR = "runtime_coordinator"
    GUARD_FAILURE = "guard_failure"
    STARTUP_RECONCILE_FAILED = "startup_reconcile_failed"
    DERIVATIVES_NOT_ACCESSIBLE = "derivatives_not_accessible"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    USER_REQUEST = "user_request"
    SYSTEM_MONITOR = "system_monitor"


@dataclass
class OriginalReduceOnlyModeState:
    """State data for reduce_only_mode tracking (compatibility wrapper)."""

    enabled: bool = False
    reason: str | None = None
    source: ReduceOnlyModeSource | None = None
    timestamp: datetime | None = None
    previous_state: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for logging/serialization."""
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "source": self.source.value if self.source else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "previous_state": self.previous_state,
        }


class StateType(Enum):
    """Enumeration of different state types managed by the unified system."""

    REDUCE_ONLY_MODE = "reduce_only_mode"
    TRADING_ENABLED = "trading_enabled"
    EMERGENCY_STOP = "emergency_stop"
    SYSTEM_STATUS = "system_status"
    CONFIGURATION = "configuration"
    SESSION_STATE = "session_state"


@dataclass
class StateChange:
    """Record of a state change for auditing."""

    timestamp: datetime
    state_type: StateType
    old_value: Any
    new_value: Any
    source: str
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "state_type": self.state_type.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "source": self.source,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class SystemState(BaseManager):
    """Unified state management system for all orchestration state.

    This consolidates state management responsibilities from multiple managers
    into a single, thread-safe, auditable system.
    """

    def __init__(self, name: str = "SystemState") -> None:
        super().__init__(name)
        self._lock = RLock()
        self._state: dict[StateType, Any] = {}
        self._change_listeners: dict[StateType, list[Callable[[StateChange], None]]] = {}
        self._change_history: list[StateChange] = []
        self._max_history_size = 1000

        # Initialize default states
        self._initialize_default_states()

    def _initialize_manager(self) -> None:
        """Initialize the unified state manager."""
        # Initialize with safe defaults
        self._update_status("initialized", "Unified state manager initialized")

    def _on_shutdown(self) -> None:
        """Cleanup on shutdown."""
        with self._lock:
            self._change_listeners.clear()
            self._change_history.clear()
            self._state.clear()

    def _initialize_default_states(self) -> None:
        """Initialize default state values."""
        # Reduce-only mode defaults to False
        self._state[StateType.REDUCE_ONLY_MODE] = False

        # Trading enabled defaults to True (unless reduce_only_mode is True)
        self._state[StateType.TRADING_ENABLED] = True

        # Emergency stop defaults to False
        self._state[StateType.EMERGENCY_STOP] = False

        # System status defaults to "starting"
        self._state[StateType.SYSTEM_STATUS] = "starting"

        # Session state defaults to "inactive"
        self._state[StateType.SESSION_STATE] = "inactive"

    # Public state access methods
    def get_state(self, state_type: StateType) -> Any:
        """Get the current value of a state type."""
        with self._lock:
            return self._state.get(state_type)

    def set_state(
        self,
        state_type: StateType,
        value: Any,
        source: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        validate_change: bool = True,
    ) -> None:
        """Set a state value with validation and audit logging.

        Args:
            state_type: The type of state to set
            value: The new value for the state
            source: Source of the state change (for auditing)
            reason: Optional reason for the change
            metadata: Optional additional metadata
            validate_change: Whether to validate the change (default: True)
        """
        self._ensure_initialized()

        with self._lock:
            old_value = self._state.get(state_type)

            if validate_change:
                try:
                    self._validate_state_change(state_type, old_value, value)
                except ValueError as exc:
                    raise ValueError(f"Invalid state change for {state_type.value}: {exc}") from exc

            # Apply the change
            self._state[state_type] = value

            # Create change record
            change = StateChange(
                timestamp=datetime.utcnow(),
                state_type=state_type,
                old_value=old_value,
                new_value=value,
                source=source,
                reason=reason,
                metadata=metadata or {},
            )

            # Record in history
            self._record_change(change)

            # Notify listeners
            self._notify_listeners(state_type, change)

            # Log the change
            from bot_v2.utilities.logging_patterns import get_logger

            logger = get_logger(__name__, component="unified_state_manager")

            logger.info(
                f"State change: {state_type.value} = {value}",
                operation="state_change",
                state_type=state_type.value,
                old_value=old_value,
                new_value=value,
                source=source,
                reason=reason,
            )

    def validate_and_set_state(
        self,
        state_type: StateType,
        value: Any,
        source: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Validate and set state, returning success status.

        This method validates the state change and only applies it if validation passes.
        Returns True if the change was applied, False otherwise.
        """
        try:
            self.set_state(state_type, value, source, reason, metadata, validate_change=True)
            return True
        except ValueError:
            return False

    # Reduce-only mode specific methods (for backward compatibility)
    def get_reduce_only_mode(self) -> bool:
        """Get the current reduce-only mode state."""
        return cast(bool, self.get_state(StateType.REDUCE_ONLY_MODE))

    def set_reduce_only_mode(
        self,
        enabled: bool,
        source: str,
        reason: str | None = None,
    ) -> None:
        """Set reduce-only mode state."""
        self.set_state(
            StateType.REDUCE_ONLY_MODE,
            enabled,
            source,
            reason,
        )

        # Automatically update trading enabled state
        if enabled:
            self.set_state(
                StateType.TRADING_ENABLED,
                False,
                source,
                (
                    f"Trading disabled due to reduce-only mode: {reason}"
                    if reason
                    else "Reduce-only mode enabled"
                ),
            )
        else:
            # Only enable trading if no other emergency conditions exist
            emergency_stop = self.get_state(StateType.EMERGENCY_STOP)
            if not emergency_stop:
                self.set_state(
                    StateType.TRADING_ENABLED,
                    True,
                    source,
                    (
                        f"Trading enabled due to reduce-only mode disabled: {reason}"
                        if reason
                        else "Reduce-only mode disabled"
                    ),
                )

    def to_legacy_state(self) -> OriginalReduceOnlyModeState:
        """Convert current state to legacy format for compatibility."""
        from datetime import datetime as dt_datetime

        return OriginalReduceOnlyModeState(
            enabled=self.get_reduce_only_mode(),
            reason=self._get_last_change_reason(StateType.REDUCE_ONLY_MODE),
            source=self._map_source_to_legacy(
                self._get_last_change_source(StateType.REDUCE_ONLY_MODE)
            ),
            timestamp=cast(
                dt_datetime, self._get_last_change_timestamp(StateType.REDUCE_ONLY_MODE)
            ),
            previous_state=self._get_previous_state(StateType.REDUCE_ONLY_MODE),
        )

    # State change listeners
    def add_state_listener(
        self,
        state_type: StateType,
        listener: Callable[[StateChange], None],
    ) -> None:
        """Add a listener for state changes of a specific type."""
        with self._lock:
            if state_type not in self._change_listeners:
                self._change_listeners[state_type] = []
            self._change_listeners[state_type].append(listener)

    def remove_state_listener(
        self,
        state_type: StateType,
        listener: Callable[[StateChange], None],
    ) -> None:
        """Remove a listener for state changes."""
        with self._lock:
            if state_type in self._change_listeners:
                try:
                    self._change_listeners[state_type].remove(listener)
                except ValueError:
                    pass  # Listener not found, ignore

    # History and audit methods
    def get_change_history(
        self,
        state_type: StateType | None = None,
        limit: int | None = None,
    ) -> list[StateChange]:
        """Get the history of state changes."""
        with self._lock:
            history = self._change_history

            if state_type is not None:
                history = [c for c in history if c.state_type == state_type]

            if limit is not None:
                history = history[-limit:]

            return history.copy()

    def clear_history(self, state_type: StateType | None = None) -> None:
        """Clear state change history."""
        with self._lock:
            if state_type is None:
                self._change_history.clear()
            else:
                self._change_history = [
                    c for c in self._change_history if c.state_type != state_type
                ]

    # State validation methods
    def _validate_state_change(self, state_type: StateType, old_value: Any, new_value: Any) -> None:
        """Validate a state change before applying it."""
        if state_type == StateType.REDUCE_ONLY_MODE:
            if not isinstance(new_value, bool):
                raise ValueError("reduce_only_mode must be a boolean")
        elif new_value and self.get_state(StateType.EMERGENCY_STOP):
            raise ValueError("Cannot enable reduce_only_mode during emergency stop")

        elif state_type == StateType.EMERGENCY_STOP:
            if not isinstance(new_value, bool):
                raise ValueError("emergency_stop must be a boolean")

        elif state_type == StateType.TRADING_ENABLED:
            if not isinstance(new_value, bool):
                raise ValueError("trading_enabled must be a boolean")
            elif new_value and self.get_state(StateType.REDUCE_ONLY_MODE):
                raise ValueError("Cannot enable trading while reduce_only_mode is enabled")

    # Internal helper methods
    def _record_change(self, change: StateChange) -> None:
        """Record a state change in history."""
        self._change_history.append(change)

        # Trim history if it gets too large
        if len(self._change_history) > self._max_history_size:
            self._change_history = self._change_history[-self._max_history_size :]

    def _notify_listeners(self, state_type: StateType, change: StateChange) -> None:
        """Notify all listeners for a state change."""
        listeners = self._change_listeners.get(state_type, [])
        for listener in listeners:
            try:
                listener(change)
            except Exception as exc:
                from bot_v2.utilities.logging_patterns import get_logger

                logger = get_logger(__name__, component="unified_state_manager")
                logger.error(
                    "State change listener error",
                    operation="state_listener_error",
                    state_type=state_type.value,
                    listener=str(listener),
                    error=str(exc),
                    exc_info=True,
                )

    def _get_last_change_source(self, state_type: StateType) -> str | None:
        """Get the source of the last change for a state type."""
        for change in reversed(self._change_history):
            if change.state_type == state_type:
                return change.source
        return None

    def _get_last_change_reason(self, state_type: StateType) -> str | None:
        """Get the reason of the last change for a state type."""
        for change in reversed(self._change_history):
            if change.state_type == state_type:
                return change.reason
        return None

    def _get_last_change_timestamp(self, state_type: StateType) -> datetime | None:
        """Get the timestamp of the last change for a state type."""
        for change in reversed(self._change_history):
            if change.state_type == state_type:
                return change.timestamp
        return None

    def _get_previous_state(self, state_type: StateType) -> Any:
        """Get the previous value for a state type."""
        found_current = False
        for change in reversed(self._change_history):
            if change.state_type == state_type:
                if found_current:
                    return change.old_value
                found_current = True
        return None  # No previous state found

    def _map_source_to_legacy(self, source: str | None) -> ReduceOnlyModeSource:
        """Map modern source names to legacy enum values."""
        if source is None:
            return ReduceOnlyModeSource.RUNTIME_COORDINATOR

        source_mapping = {
            "config": ReduceOnlyModeSource.CONFIG,
            "risk_manager": ReduceOnlyModeSource.RISK_MANAGER,
            "runtime_coordinator": ReduceOnlyModeSource.RUNTIME_COORDINATOR,
            "guard_failure": ReduceOnlyModeSource.GUARD_FAILURE,
            "user_request": ReduceOnlyModeSource.USER_REQUEST,
            "system_monitor": ReduceOnlyModeSource.SYSTEM_MONITOR,
            "daily_loss_limit": ReduceOnlyModeSource.DAILY_LOSS_LIMIT,
            "derivatives_not_accessible": ReduceOnlyModeSource.DERIVATIVES_NOT_ACCESSIBLE,
            "startup_reconcile_failed": ReduceOnlyModeSource.STARTUP_RECONCILE_FAILED,
        }

        return source_mapping.get(source.lower(), ReduceOnlyModeSource.RUNTIME_COORDINATOR)


def create_reduce_only_state_manager(
    event_store: Any = None,
    initial_state: bool = False,
    validation_enabled: bool = True,
) -> SystemState:
    """Create a configured SystemState instance for reduce-only mode management.

    Args:
        event_store: Optional event store for persistence
        initial_state: Initial reduce-only mode state
        validation_enabled: Whether to enable validation

    Returns:
        Configured SystemState instance
    """
    manager = SystemState(name="ReduceOnlyStateManager")

    # Set initial state
    manager.set_reduce_only_mode(
        enabled=initial_state,
        source="system_initialization",
        reason="Initial state configuration",
    )

    return manager


__all__ = [
    "StateType",
    "StateChange",
    "SystemState",
    "create_reduce_only_state_manager",
]

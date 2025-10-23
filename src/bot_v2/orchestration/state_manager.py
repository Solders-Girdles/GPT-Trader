"""
Centralized state manager for reduce_only_mode mutations.

This module implements a centralized StateManager that controls all mutations
to the reduce_only_mode state, providing validation, audit logging, and
consistent behavior across the application.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from bot_v2.persistence.event_store import EventStore
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.telemetry import emit_metric

logger = get_logger(__name__, component="state_manager")


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
class ReduceOnlyModeState:
    """State data for reduce_only_mode tracking."""

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


@dataclass
class StateChangeRequest:
    """Request to change reduce_only_mode state."""

    enabled: bool
    reason: str
    source: ReduceOnlyModeSource
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class ReduceOnlyModeStateManager:
    """
    Centralized state manager for reduce_only_mode mutations.

    This class provides a single point of control for all reduce_only_mode
    mutations, ensuring consistent validation, logging, and state tracking.
    It replaces the distributed mutation points that were previously located
    in ConfigController, StateManager, and RuntimeCoordinator.
    """

    def __init__(
        self,
        event_store: EventStore,
        now_provider: Callable[[], datetime] | None = None,
        initial_state: bool = False,
        validation_enabled: bool = True,
    ):
        """
        Initialize the state manager.

        Args:
            event_store: Event store for persisting state changes
            now_provider: Time provider for testability
            initial_state: Initial reduce_only_mode state
            validation_enabled: Whether to enable strict validation
        """
        self.event_store = event_store
        self._now_provider = now_provider or (lambda: datetime.utcnow())
        self._validation_enabled = validation_enabled

        # Current state
        self._state = ReduceOnlyModeState(enabled=initial_state)

        # State change listeners
        self._state_listeners: list[Callable[[ReduceOnlyModeState], None]] = []

        # Audit log of all changes
        self._audit_log: list[StateChangeRequest] = []

        logger.info(
            "ReduceOnlyModeStateManager initialized",
            operation="state_manager_init",
            initial_state=initial_state,
            validation_enabled=validation_enabled,
        )

    @property
    def current_state(self) -> ReduceOnlyModeState:
        """Get the current state."""
        return self._state

    @property
    def is_reduce_only_mode(self) -> bool:
        """Check if reduce_only_mode is currently enabled."""
        return self._state.enabled

    def set_reduce_only_mode(
        self,
        enabled: bool,
        reason: str,
        source: ReduceOnlyModeSource,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Set the reduce_only_mode state with validation and logging.

        This is the single point of mutation for reduce_only_mode state.
        All other components should use this method to change the state.

        Args:
            enabled: Whether to enable reduce_only_mode
            reason: Reason for the state change
            source: Source of the state change
            metadata: Additional metadata about the change

        Returns:
            True if the state was changed, False if no change was needed

        Raises:
            ValueError: If validation fails and validation is enabled
        """
        # Check if state is already the same
        if enabled == self._state.enabled:
            logger.debug(
                "Reduce-only mode state unchanged",
                operation="reduce_only_mode_set",
                enabled=enabled,
                reason=reason,
                source=source.value,
                current_state=self._state.enabled,
            )
            return False

        # Create state change request
        request = StateChangeRequest(
            enabled=enabled,
            reason=reason,
            source=source,
            metadata=metadata or {},
        )

        # Validate the request
        if self._validation_enabled:
            self._validate_state_change(request)

        # Record the previous state
        previous_state = self._state.enabled

        # Update the state
        self._state = ReduceOnlyModeState(
            enabled=enabled,
            reason=reason,
            source=source,
            timestamp=request.timestamp,
            previous_state=previous_state,
        )

        # Add to audit log
        self._audit_log.append(request)

        # Emit metrics and log the change
        self._emit_state_change_metric(request)
        self._log_state_change(request, previous_state)

        # Notify listeners
        self._notify_state_listeners()

        logger.info(
            "Reduce-only mode state changed",
            operation="reduce_only_mode_changed",
            enabled=enabled,
            reason=reason,
            source=source.value,
            previous_state=previous_state,
            timestamp=request.timestamp.isoformat(),
        )

        return True

    def _validate_state_change(self, request: StateChangeRequest) -> None:
        """
        Validate a state change request.

        Args:
            request: The state change request to validate

        Raises:
            ValueError: If validation fails
        """
        # Validate reason is not empty
        if not request.reason or not request.reason.strip():
            raise ValueError("Reason cannot be empty when changing reduce_only_mode")

        # Validate source is appropriate for the change
        if request.source == ReduceOnlyModeSource.CONFIG and not request.enabled:
            # Config source should only be used for enabling, not disabling
            logger.warning(
                "Config source used to disable reduce_only_mode, consider using a more specific source",
                operation="reduce_only_mode_validation",
                source=request.source.value,
                enabled=request.enabled,
            )

        # Validate metadata for specific sources
        if request.source == ReduceOnlyModeSource.DAILY_LOSS_LIMIT:
            if "loss_amount" not in request.metadata:
                raise ValueError("Daily loss limit source requires 'loss_amount' in metadata")

    def _emit_state_change_metric(self, request: StateChangeRequest) -> None:
        """Emit a metric for the state change."""
        try:
            emit_metric(
                self.event_store,
                "state_manager",
                {
                    "event_type": "reduce_only_mode_changed",
                    "enabled": request.enabled,
                    "reason": request.reason,
                    "source": request.source.value,
                    "timestamp": request.timestamp.isoformat(),
                    "metadata": request.metadata,
                },
                logger=logger,
            )
        except Exception as exc:
            logger.warning(
                "Failed to emit reduce_only_mode state change metric",
                operation="reduce_only_mode_metric",
                error=str(exc),
                exc_info=True,
            )

    def _log_state_change(self, request: StateChangeRequest, previous_state: bool) -> None:
        """Log the state change with appropriate level."""
        message = (
            f"Reduce-only mode {'enabled' if request.enabled else 'disabled'} "
            f"by {request.source.value}: {request.reason}"
        )

        if request.source in {
            ReduceOnlyModeSource.GUARD_FAILURE,
            ReduceOnlyModeSource.STARTUP_RECONCILE_FAILED,
            ReduceOnlyModeSource.DERIVATIVES_NOT_ACCESSIBLE,
        }:
            # These are critical events that should be logged as warnings
            logger.warning(
                message,
                operation="reduce_only_mode_critical_change",
                source=request.source.value,
                reason=request.reason,
                previous_state=previous_state,
                new_state=request.enabled,
                metadata=request.metadata,
            )
        else:
            # Regular informational logging
            logger.info(
                message,
                operation="reduce_only_mode_change",
                source=request.source.value,
                reason=request.reason,
                previous_state=previous_state,
                new_state=request.enabled,
                metadata=request.metadata,
            )

    def _notify_state_listeners(self) -> None:
        """Notify all registered state listeners."""
        for listener in self._state_listeners:
            try:
                listener(self._state)
            except Exception as exc:
                logger.warning(
                    "State listener failed",
                    operation="state_listener_error",
                    error=str(exc),
                    exc_info=True,
                )

    def add_state_listener(self, listener: Callable[[ReduceOnlyModeState], None]) -> None:
        """
        Add a listener for state changes.

        Args:
            listener: Callback function to be called when state changes
        """
        self._state_listeners.append(listener)
        logger.debug(
            "State listener added",
            operation="state_listener_added",
            listener_count=len(self._state_listeners),
        )

    def remove_state_listener(self, listener: Callable[[ReduceOnlyModeState], None]) -> None:
        """
        Remove a state change listener.

        Args:
            listener: Callback function to remove
        """
        try:
            self._state_listeners.remove(listener)
            logger.debug(
                "State listener removed",
                operation="state_listener_removed",
                listener_count=len(self._state_listeners),
            )
        except ValueError:
            logger.warning(
                "Attempted to remove non-existent state listener",
                operation="state_listener_remove_error",
            )

    def get_audit_log(self, limit: int | None = None) -> list[StateChangeRequest]:
        """
        Get the audit log of state changes.

        Args:
            limit: Maximum number of entries to return (most recent first)

        Returns:
            List of state change requests
        """
        if limit is None:
            return list(reversed(self._audit_log))
        return list(reversed(self._audit_log[-limit:]))

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        count = len(self._audit_log)
        self._audit_log.clear()
        logger.info(
            "Audit log cleared",
            operation="audit_log_cleared",
            entries_cleared=count,
        )

    def reset_state(self, enabled: bool = False, reason: str = "reset") -> bool:
        """
        Reset the state to a known value.

        Args:
            enabled: The new state value
            reason: Reason for the reset

        Returns:
            True if the state was changed
        """
        return self.set_reduce_only_mode(
            enabled=enabled,
            reason=reason,
            source=ReduceOnlyModeSource.SYSTEM_MONITOR,
            metadata={"operation": "reset"},
        )

    def get_state_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current state and history.

        Returns:
            Dictionary with state summary information
        """
        return {
            "current_state": self._state.to_dict(),
            "change_count": len(self._audit_log),
            "last_change": self._audit_log[-1].timestamp.isoformat() if self._audit_log else None,
            "listener_count": len(self._state_listeners),
            "validation_enabled": self._validation_enabled,
        }


# Convenience functions for creating state manager instances
def create_reduce_only_state_manager(
    event_store: EventStore,
    initial_state: bool = False,
    validation_enabled: bool = True,
) -> ReduceOnlyModeStateManager:
    """
    Create a ReduceOnlyModeStateManager with default settings.

    Args:
        event_store: Event store for persisting state changes
        initial_state: Initial reduce_only_mode state
        validation_enabled: Whether to enable strict validation

    Returns:
        Configured ReduceOnlyModeStateManager instance
    """
    return ReduceOnlyModeStateManager(
        event_store=event_store,
        initial_state=initial_state,
        validation_enabled=validation_enabled,
    )


__all__ = [
    "ReduceOnlyModeStateManager",
    "ReduceOnlyModeState",
    "ReduceOnlyModeSource",
    "StateChangeRequest",
    "create_reduce_only_state_manager",
]

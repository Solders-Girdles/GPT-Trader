"""
State registry for efficient state propagation to widgets.

This module provides a centralized registry for widgets that need to receive
TuiState updates. It replaces the manual query-based approach in MainScreen
with a more scalable broadcast pattern.

Pattern based on TuiLogHandler in log_manager.py.
Includes performance instrumentation for monitoring broadcast timing.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol
from weakref import WeakSet

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")

# Threshold for slow broadcast warning (seconds)
SLOW_BROADCAST_THRESHOLD = 0.020  # 20ms


class StateObserver(Protocol):
    """Protocol for widgets that observe TuiState changes.

    Widgets implementing this protocol can register with StateRegistry
    to receive state updates via on_state_updated().
    """

    def on_state_updated(self, state: TuiState) -> None:
        """Called when TuiState is updated.

        Args:
            state: The updated TuiState instance.
        """
        ...


class StateRegistry:
    """Registry for widgets that need state updates.

    Provides a broadcast mechanism for propagating TuiState changes
    to all registered observers. Uses WeakSet for automatic cleanup
    when widgets are garbage collected.

    Usage:
        # In app initialization:
        self.state_registry = StateRegistry()

        # In widget on_mount:
        self.app.state_registry.register(self)

        # In widget on_unmount:
        self.app.state_registry.unregister(self)

        # In MainScreen.watch_state:
        self.app.state_registry.broadcast(state)

    Attributes:
        _observers: WeakSet of registered state observers.
    """

    def __init__(self) -> None:
        """Initialize the state registry with empty observer set."""
        self._observers: WeakSet[StateObserver] = WeakSet()

    def register(self, observer: StateObserver) -> None:
        """Register a widget to receive state updates.

        Args:
            observer: Widget implementing StateObserver protocol.
        """
        self._observers.add(observer)
        logger.debug(f"Registered state observer: {type(observer).__name__}")

    def unregister(self, observer: StateObserver) -> None:
        """Unregister a widget from state updates.

        Args:
            observer: Widget to unregister.
        """
        self._observers.discard(observer)
        logger.debug(f"Unregistered state observer: {type(observer).__name__}")

    def broadcast(self, state: TuiState) -> None:
        """Broadcast state update to all registered observers.

        Each observer is updated in a try/except to prevent one failing
        widget from blocking updates to others.

        Includes performance monitoring for slow broadcasts.

        Args:
            state: The TuiState to broadcast.
        """
        # Import here to avoid circular imports
        from gpt_trader.tui.services.performance_service import (
            get_tui_performance_service,
            perf_trace,
        )

        perf = get_tui_performance_service()
        start_time = time.time()

        # Convert to list to avoid iteration issues if observers are modified
        observers = list(self._observers)
        notified_count = 0

        for observer in observers:
            try:
                observer.on_state_updated(state)
                notified_count += 1
            except Exception as e:
                # Log but don't fail - one widget shouldn't block others
                logger.warning(
                    f"Error broadcasting state to {type(observer).__name__}: {e}",
                    exc_info=True,
                )

        # Record slow broadcasts for performance monitoring
        duration = time.time() - start_time
        if duration > SLOW_BROADCAST_THRESHOLD:
            perf.record_slow_operation(f"broadcast({notified_count})", duration)

        # Trace broadcast timing
        perf_trace(
            "StateRegistry.broadcast",
            duration * 1000,
            observers=notified_count,
        )

    @property
    def observer_count(self) -> int:
        """Number of currently registered observers.

        Returns:
            Count of registered observers.
        """
        return len(self._observers)

    def clear(self) -> None:
        """Clear all registered observers.

        Useful for cleanup during app shutdown.
        """
        self._observers.clear()
        logger.debug("Cleared all state observers")

"""Status mixin for TraderApp.

Contains methods related to status updates, observer connections,
and state synchronization.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from gpt_trader.monitoring.status_reporter import BotStatus
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp

logger = get_logger(__name__, component="tui")


class TraderAppStatusMixin:
    """Mixin providing status methods for TraderApp.

    Methods:
    - _on_status_update: Callback for StatusReporter updates
    - _bind_state: Bind reactive state to widgets
    - _is_real_status_reporter: Check if real reporter (not null)
    - connect_status_observer: Connect StatusReporter observer
    - _sync_state_from_bot: Manually sync state from bot
    - _pulse_heartbeat: Smooth heartbeat pulse animation
    """

    # Type hints for attributes from TraderApp
    if TYPE_CHECKING:
        bot: Any
        tui_state: Any
        ui_coordinator: Any

        def query_one(self, selector: str) -> Any: ...
        def call_from_thread(self, callback: Any, *args: Any) -> None: ...

    def _on_status_update(self: TraderApp, status: BotStatus) -> None:
        """Callback for StatusReporter updates (receives typed BotStatus).

        Delegates to UICoordinator, using call_from_thread only when called
        from a background thread (not from the main asyncio event loop thread).
        """
        if self.ui_coordinator:
            # Check if we're on the main thread (where asyncio event loop runs)
            if threading.current_thread() is threading.main_thread():
                # Already on main thread - call directly
                self.ui_coordinator.apply_observer_update(status)
            else:
                # On background thread - use call_from_thread for thread safety
                self.call_from_thread(self.ui_coordinator.apply_observer_update, status)

    def _bind_state(self: TraderApp) -> None:
        """Bind reactive state to widgets."""
        # This is where we could set up direct bindings if widgets supported it
        # For now, we'll just rely on the update loop pushing data to state,
        # and then we can push state to widgets or have widgets watch state.
        pass

    def _is_real_status_reporter(self: TraderApp) -> bool:
        """Check if the current status_reporter is a real reporter (not NullStatusReporter).

        Returns:
            True if it's a real reporter that provides data, False if it's a null adapter.
        """
        if not self.bot or not hasattr(self.bot.engine, "status_reporter"):
            return False
        reporter = self.bot.engine.status_reporter
        return not getattr(reporter, "is_null_reporter", False)

    def connect_status_observer(self: TraderApp) -> None:
        """Connect the StatusReporter observer callback.

        Called by MainScreen.on_mount() after widgets are mounted to avoid
        race condition where status updates arrive before widgets are ready.

        Skips connection for NullStatusReporter (degraded mode) since it
        doesn't push updates.
        """
        if not self.bot or not hasattr(self.bot.engine, "status_reporter"):
            logger.warning("Cannot connect observer: bot or status_reporter not available")
            return

        # Skip observer connection for NullStatusReporter (it doesn't push updates)
        if not self._is_real_status_reporter():
            logger.debug("Skipping observer connection for NullStatusReporter (degraded mode)")
            return

        if hasattr(self.bot, "set_ui_adapter"):
            from gpt_trader.tui.adapters.runtime_ui_adapter import TuiRuntimeUIAdapter

            self.bot.set_ui_adapter(TuiRuntimeUIAdapter(self))
            logger.debug("Connected runtime UI adapter (from MainScreen.on_mount)")
            return

        self.bot.engine.status_reporter.add_observer(self._on_status_update)
        logger.debug("Connected to StatusReporter observer (from MainScreen.on_mount)")

    def _sync_state_from_bot(self: TraderApp) -> None:
        """Manually sync state from bot (delegates to UICoordinator when available).

        Called by BotLifecycleManager on bot start/stop/mode-switch.
        Falls back to direct state update when ui_coordinator is None (e.g., in tests).
        Handles NullStatusReporter gracefully in degraded mode.
        """
        if self.ui_coordinator:
            self.ui_coordinator.sync_state_from_bot()
        elif self.bot:
            # Fallback for tests and pre-mount scenarios
            self.tui_state.running = self.bot.running

            # Access runtime state safely
            runtime_state = None
            if hasattr(self.bot, "engine") and hasattr(self.bot.engine, "context"):
                if hasattr(self.bot.engine.context, "runtime_state"):
                    runtime_state = self.bot.engine.context.runtime_state

            # Access StatusReporter for typed data (skip for NullStatusReporter)
            if hasattr(self.bot, "engine") and hasattr(self.bot.engine, "status_reporter"):
                # Check if this is a NullStatusReporter (degraded mode)
                if not self._is_real_status_reporter():
                    logger.debug("Skipping status sync for NullStatusReporter (degraded mode)")
                    self.tui_state.connection_healthy = False
                    return

                status = self.bot.engine.status_reporter.get_status()
                self.tui_state.update_from_bot_status(status, runtime_state)

    def _pulse_heartbeat(self: TraderApp) -> None:
        """Smooth heartbeat pulse using sine wave."""
        import math
        import time

        from gpt_trader.tui.widgets.status import BotStatusWidget

        try:
            status_widget = self.query_one(BotStatusWidget)

            # Calculate sine wave: 0.0 to 1.0
            t = time.time()
            pulse = (math.sin(t * 2) + 1) / 2  # Sine wave normalized to 0-1

            status_widget.heartbeat = pulse
        except Exception as e:
            logger.debug(f"Failed to pulse heartbeat: {e}")

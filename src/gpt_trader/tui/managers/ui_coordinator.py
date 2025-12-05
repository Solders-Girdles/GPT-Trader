"""
UI Coordinator for TUI.

Handles UI update orchestration from bot status updates via observer callbacks.
Provides heartbeat animation loop. Extracted from TraderApp to reduce complexity.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from gpt_trader.monitoring.status_reporter import BotStatus
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp

logger = get_logger(__name__, component="tui")


class UICoordinator:
    """
    Coordinates UI updates from bot status changes.

    Data updates are driven by observer callbacks (apply_observer_update).
    Heartbeat loop provides visual animation (no data polling).
    """

    def __init__(self, app: TraderApp):
        """
        Initialize UICoordinator.

        Args:
            app: Reference to the TraderApp instance
        """
        self.app = app
        self._update_task: asyncio.Task | None = None  # Heartbeat task

    def apply_observer_update(self, status: BotStatus) -> None:
        """
        Apply typed status update to state and UI.

        Called from observer callback on bot status changes. Protected with
        error boundaries to prevent cascade failures.

        Args:
            status: Typed BotStatus snapshot from StatusReporter
        """
        try:
            logger.debug(
                f"Applying status update to TUI state. Bot running: {self.app.bot.running}"
            )

            # Get runtime state
            runtime_state = None
            if hasattr(self.app.bot.engine.context, "runtime_state"):
                runtime_state = self.app.bot.engine.context.runtime_state
                logger.debug(
                    f"Runtime state available: uptime={getattr(runtime_state, 'uptime', 'N/A')}"
                )

            # Update TuiState (critical operation - isolated error handling)
            try:
                self.app.tui_state.running = self.app.bot.running
                self.app.tui_state.update_from_bot_status(status, runtime_state)

                # Log successful update
                logger.debug(
                    f"State updated successfully: "
                    f"positions={len(status.positions.positions) if status.positions else 0}, "
                    f"market_symbols={len(status.market.last_prices) if status.market else 0}"
                )
            except Exception as e:
                logger.error(f"Failed to update TuiState from bot status: {e}", exc_info=True)
                self.app.notify(
                    "Critical: State update failed. UI may show stale data.",
                    title="State Update Error",
                    severity="error",
                    timeout=10,
                )
                # Don't proceed to UI update if state update failed
                return

            # Update UI (already has some error handling in update_main_screen)
            self.update_main_screen()

        except Exception as e:
            logger.error(f"Critical error in apply_observer_update: {e}", exc_info=True)
            # Last resort notification
            try:
                self.app.notify(
                    "Critical UI update failure. Consider restarting TUI.",
                    title="Critical Error",
                    severity="error",
                    timeout=30,
                )
            except Exception:
                # Even notification failed - log and continue
                logger.critical("Cannot notify user of critical error - TUI may be unresponsive")

    def sync_state_from_bot(self) -> None:
        """
        Manually sync state from bot (for reconnect action).

        Fetches current status from StatusReporter and updates UI.
        """
        self.app.tui_state.running = self.app.bot.running

        # Access runtime state safely
        runtime_state = None
        if hasattr(self.app.bot.engine.context, "runtime_state"):
            runtime_state = self.app.bot.engine.context.runtime_state

        # Access StatusReporter for typed data
        if hasattr(self.app.bot.engine, "status_reporter"):
            status = self.app.bot.engine.status_reporter.get_status()  # Returns BotStatus
            logger.debug(
                f"Fetched status from StatusReporter: bot_id={status.bot_id}, "
                f"timestamp={status.timestamp_iso}"
            )
            self.app.tui_state.update_from_bot_status(status, runtime_state)
        else:
            logger.debug("No StatusReporter available, skipping status update")

    def update_main_screen(self) -> None:
        """Update the main screen UI with current state."""
        try:
            from gpt_trader.tui.screens import MainScreen
            from gpt_trader.tui.screens.main import SystemDetailsScreen

            main_screen = self.app.query_one(MainScreen)
            main_screen.update_ui(self.app.tui_state)

            # Propagate state to SystemDetailsScreen if it's showing
            try:
                system_details = self.app.query_one(SystemDetailsScreen)
                system_details.state = self.app.tui_state
            except Exception:
                # SystemDetailsScreen not mounted - that's fine
                pass

            # Toggle heartbeat to show dashboard is alive
            self.app._pulse_heartbeat()
            logger.debug("UI updated successfully")
        except Exception as e:
            logger.warning(f"Failed to update main screen from status update: {e}")

    async def start_update_loop(self) -> None:
        """
        Start the periodic heartbeat loop.

        This loop runs every second to pulse the heartbeat animation.
        Data updates are handled by the observer callback (apply_observer_update).
        """
        logger.info("Starting UI heartbeat loop")
        self._update_task = asyncio.create_task(self._heartbeat_loop())

    async def stop_update_loop(self) -> None:
        """Stop the periodic heartbeat loop."""
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                logger.info("Heartbeat loop cancelled successfully")
            self._update_task = None
            logger.info("UI heartbeat loop stopped")

    async def _heartbeat_loop(self) -> None:
        """
        Periodic heartbeat loop (runs every 1 second).

        Only pulses the heartbeat animation to show the dashboard is alive.
        Data updates are handled by the observer callback (no polling needed).
        """
        loop_count = 0
        while True:
            try:
                loop_count += 1
                if loop_count % 30 == 0:  # Log every 30 seconds
                    logger.debug(f"Heartbeat loop iteration {loop_count}")

                # Only pulse heartbeat - no state sync or UI update
                self.app._pulse_heartbeat()

            except Exception as e:
                logger.debug(f"Heartbeat pulse error: {e}")

            await asyncio.sleep(1)

    def cleanup(self) -> None:
        """Clean up heartbeat tasks on manager destruction."""
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            logger.info("UICoordinator cleaned up heartbeat task")

"""
UI Coordinator for TUI.

Handles UI update orchestration from bot status updates.
Extracted from TraderApp to reduce complexity and improve testability.
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
    """Coordinates UI updates from bot status changes."""

    def __init__(self, app: TraderApp):
        """
        Initialize UICoordinator.

        Args:
            app: Reference to the TraderApp instance
        """
        self.app = app
        self._update_task: asyncio.Task | None = None

    def apply_observer_update(self, status: BotStatus) -> None:
        """
        Apply typed status update to state and UI.

        Called from observer callback on bot status changes.

        Args:
            status: Typed BotStatus snapshot from StatusReporter
        """
        logger.debug(f"Applying status update to TUI state. Bot running: {self.app.bot.running}")

        # Get runtime state
        runtime_state = None
        if hasattr(self.app.bot.engine.context, "runtime_state"):
            runtime_state = self.app.bot.engine.context.runtime_state
            logger.debug(
                f"Runtime state available: uptime={getattr(runtime_state, 'uptime', 'N/A')}"
            )

        # Update TuiState
        self.app.tui_state.running = self.app.bot.running
        self.app.tui_state.update_from_bot_status(status, runtime_state)

        # Log key data points for debugging
        if status.market:
            logger.debug(f"Market data: {len(status.market.last_prices)} symbols")
        if status.positions:
            logger.debug(f"Position data received: {status.positions.count} positions")

        # Update UI
        self.update_main_screen()

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

            main_screen = self.app.query_one(MainScreen)
            main_screen.update_ui(self.app.tui_state)
            # Toggle heartbeat to show dashboard is alive
            self.app._pulse_heartbeat()
            logger.debug("UI updated successfully")
        except Exception as e:
            logger.warning(f"Failed to update main screen from status update: {e}")

    async def start_update_loop(self) -> None:
        """
        Start the periodic update loop.

        This loop runs every second to:
        1. Sync state from bot (fallback to observer)
        2. Update UI
        3. Pulse heartbeat animation
        """
        logger.info("Starting UI update loop")
        self._update_task = asyncio.create_task(self._update_loop())

    async def stop_update_loop(self) -> None:
        """Stop the periodic update loop."""
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                logger.info("Update loop cancelled successfully")
            self._update_task = None
            logger.info("UI update loop stopped")

    async def _update_loop(self) -> None:
        """
        Periodic update loop (runs every 1 second).

        This provides:
        - Fallback if observer updates fail
        - Regular heartbeat animation
        - Consistent UI refresh rate
        """
        loop_count = 0
        while True:
            try:
                loop_count += 1
                if loop_count % 10 == 0:  # Log every 10 seconds
                    logger.debug(f"Update loop iteration {loop_count}")

                # Sync state from bot
                self.sync_state_from_bot()

                # Update UI
                self.update_main_screen()

            except Exception as e:
                logger.error(f"UI Update Loop Error: {e}", exc_info=True)

            await asyncio.sleep(1)

    def cleanup(self) -> None:
        """Clean up update tasks on manager destruction."""
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            logger.info("UICoordinator cleaned up update task")

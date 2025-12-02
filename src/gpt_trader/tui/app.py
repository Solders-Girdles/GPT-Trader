"""
Main TUI Application for GPT-Trader.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import App

from gpt_trader.tui.screens import MainScreen
from gpt_trader.tui.widgets import ConfigModal
from gpt_trader.tui.widgets.status import BotStatusWidget
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.orchestration.trading_bot.bot import TradingBot

logger = get_logger(__name__, component="tui")


class TraderApp(App):
    """GPT-Trader Terminal User Interface."""

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "toggle_bot", "Start/Stop Bot"),
        ("c", "show_config", "Config"),
        ("l", "focus_logs", "Focus Logs"),
        ("p", "panic", "PANIC"),
    ]

    def __init__(self, bot: TradingBot) -> None:
        super().__init__()
        self.bot = bot
        self._bot_task: asyncio.Task | None = None  # Track bot task for proper cancellation
        self._update_task: asyncio.Task | None = None

        # Initialize State
        from gpt_trader.tui.state import TuiState

        self.tui_state = TuiState()

    async def on_mount(self) -> None:
        """Called when app starts."""
        try:
            logger.info("TUI mounting, initializing components")
            self.push_screen(MainScreen())
            self.log("TUI Started")

            # Connect to StatusReporter observer if available
            if hasattr(self.bot.engine, "status_reporter"):
                self.bot.engine.status_reporter.add_observer(self._on_status_update)
                logger.info("Connected to StatusReporter observer")
            else:
                logger.warning("StatusReporter not available, using polling only")

            # DON'T auto-start the bot - let user press 's' to start when ready
            logger.info("Bot initialized in STOPPED state. Press 's' to start.")

            # Start UI update loop (fallback/backup and for fields not in status reporter)
            self._update_task = asyncio.create_task(self._update_loop())
            logger.info("UI update loop started")

            # Bind state to widgets
            self._bind_state()

            # Initial UI sync will happen in MainScreen.on_mount() after widgets are ready
            logger.info("TUI mounted successfully")
        except Exception as e:
            logger.critical(f"Failed to mount TUI: {e}", exc_info=True)
            self.notify(f"TUI initialization failed: {e}", severity="error", timeout=30)
            raise

    async def on_unmount(self) -> None:
        """Called when app stops."""
        try:
            logger.info("TUI unmounting, cleaning up observers")
            if hasattr(self.bot.engine, "status_reporter"):
                self.bot.engine.status_reporter.remove_observer(self._on_status_update)
                logger.info("Removed StatusReporter observer")
            logger.info("TUI unmounted successfully")
        except Exception as e:
            logger.error(f"Error during TUI unmount: {e}", exc_info=True)

    def _on_status_update(self, status: dict) -> None:
        """Callback for StatusReporter updates."""
        logger.debug(
            f"StatusReporter update received with {len(status)} keys: {list(status.keys())}"
        )
        # This might be called from a background thread or loop
        # Schedule the update on the main thread
        self.call_from_thread(self._apply_status_update, status)

    def _apply_status_update(self, status: dict) -> None:
        """Apply status update to state and UI."""
        logger.debug(f"Applying status update to TUI state. Bot running: {self.bot.running}")

        # Update State
        runtime_state = None
        if hasattr(self.bot.engine.context, "runtime_state"):
            runtime_state = self.bot.engine.context.runtime_state
            logger.debug(
                f"Runtime state available: uptime={getattr(runtime_state, 'uptime', 'N/A')}"
            )

        self.tui_state.running = self.bot.running
        self.tui_state.update_from_bot_status(status, runtime_state)

        # Log key data points for debugging
        if status.get("market"):
            logger.debug(f"Market data: {len(status['market'].get('last_prices', {}))} symbols")
        if status.get("positions"):
            logger.debug("Position data received")

        # Update UI
        try:
            main_screen = self.query_one(MainScreen)
            main_screen.update_ui(self.tui_state)
            # Toggle heartbeat to show dashboard is alive
            self._pulse_heartbeat()
            logger.debug("UI updated successfully")
        except Exception as e:
            logger.warning(f"Failed to update main screen from status update: {e}")

    def _bind_state(self) -> None:
        """Bind reactive state to widgets."""
        # This is where we could set up direct bindings if widgets supported it
        # For now, we'll just rely on the update loop pushing data to state,
        # and then we can push state to widgets or have widgets watch state.
        # To keep it simple for this refactor, we will manually update widgets
        # from state in _update_ui, but the source of truth is now self.tui_state
        pass

    def _pulse_heartbeat(self) -> None:
        """Toggle heartbeat indicator to show dashboard is alive."""
        try:
            from gpt_trader.tui.widgets.status import BotStatusWidget

            status_widget = self.query_one(BotStatusWidget)
            # Toggle the heartbeat state to create a visual pulse
            status_widget.heartbeat = not status_widget.heartbeat
        except Exception as e:
            logger.debug(f"Failed to pulse heartbeat: {e}")

    async def _update_loop(self) -> None:
        """Periodically update UI from bot state (Fallback loop)."""
        loop_count = 0
        while True:
            try:
                loop_count += 1
                # If we don't have observers, we must poll
                if (
                    not hasattr(self.bot.engine, "status_reporter")
                    or not self.bot.engine.status_reporter._observers
                ):
                    if loop_count % 10 == 0:  # Log every 10 seconds
                        logger.debug(
                            f"Polling bot state (no observers). Bot running: {self.bot.running}"
                        )
                    self._sync_state_from_bot()
                    try:
                        main_screen = self.query_one(MainScreen)
                        main_screen.update_ui(self.tui_state)
                        # Pulse heartbeat to show dashboard is alive
                        self._pulse_heartbeat()
                    except Exception as e:
                        logger.warning(f"Failed to update main screen from polling: {e}")
                else:
                    if loop_count % 30 == 0:  # Log every 30 seconds
                        observer_count = len(self.bot.engine.status_reporter._observers)
                        logger.debug(f"Observer pattern active, {observer_count} observers")
            except Exception as e:
                self.log(f"UI Update Error: {e}")
                logger.error(f"UI Update Loop Error: {e}", exc_info=True)
            await asyncio.sleep(1)

    def _sync_state_from_bot(self) -> None:
        """Fetch state from bot and update TuiState."""
        self.tui_state.running = self.bot.running

        # Access runtime state safely
        runtime_state = None
        if hasattr(self.bot.engine.context, "runtime_state"):
            runtime_state = self.bot.engine.context.runtime_state

        # Access StatusReporter for data
        status = {}
        if hasattr(self.bot.engine, "status_reporter"):
            status = self.bot.engine.status_reporter.get_status()
            logger.debug(f"Fetched status from StatusReporter: {len(status)} keys")
        else:
            logger.debug("No StatusReporter available, using empty status")

        self.tui_state.update_from_bot_status(status, runtime_state)

    async def action_toggle_bot(self) -> None:
        """Toggle bot running state."""
        try:
            if self.bot.running and self._bot_task:
                # Stop the bot with proper task cancellation
                self.notify("Stopping bot...", title="Status")
                logger.info("User initiated bot stop via TUI")

                # Cancel the running task
                self._bot_task.cancel()
                try:
                    await self._bot_task
                except asyncio.CancelledError:
                    logger.info("Bot task cancelled successfully")

                # Clean up task reference
                self._bot_task = None

                # Ensure bot.stop() is called for cleanup
                await self.bot.stop()

                # Immediately sync state to UI
                self._sync_state_from_bot()
                try:
                    main_screen = self.query_one(MainScreen)
                    main_screen.update_ui(self.tui_state)
                    self._pulse_heartbeat()
                    logger.info("UI state synced immediately after bot stop")
                except Exception as e:
                    logger.warning(f"Failed to sync UI state after bot stop: {e}")

                self.notify("Bot stopped.", title="Status", severity="information")
                logger.info("Bot stopped successfully via TUI")

            elif not self.bot.running:
                # Start the bot and store task reference
                self.notify("Starting bot...", title="Status")
                logger.info("User initiated bot start via TUI")

                # Store task reference for future cancellation
                self._bot_task = asyncio.create_task(self.bot.run())

                # Give it a moment to start
                await asyncio.sleep(0.2)

                # Immediately sync state to UI
                self._sync_state_from_bot()
                try:
                    main_screen = self.query_one(MainScreen)
                    main_screen.update_ui(self.tui_state)
                    self._pulse_heartbeat()
                    logger.info("UI state synced immediately after bot start")
                except Exception as e:
                    logger.warning(f"Failed to sync UI state after bot start: {e}")

                self.notify("Bot started.", title="Status", severity="information")
                logger.info("Bot started successfully via TUI")
            else:
                # Edge case: bot says it's running but we don't have task reference
                logger.warning("Bot state inconsistent: running=True but no task reference")
                self.notify("Bot state inconsistent, attempting recovery...", severity="warning")
                await self.bot.stop()
                self._bot_task = None

        except Exception as e:
            logger.error(f"Failed to toggle bot state: {e}", exc_info=True)
            self.notify(f"Error toggling bot: {e}", severity="error", timeout=10)
            # Clean up on error
            self._bot_task = None

    async def action_show_config(self) -> None:
        """Show configuration modal."""
        try:
            self.push_screen(ConfigModal(self.bot.config))
            logger.debug("Config modal opened")
        except Exception as e:
            logger.error(f"Failed to show config modal: {e}", exc_info=True)
            self.notify(f"Error showing config: {e}", severity="error")

    async def action_focus_logs(self) -> None:
        """Focus the log widget."""
        # Focus the full log widget and switch tab if needed
        try:
            self.query_one("#logs-full").focus()
        except Exception as e:
            logger.warning(f"Failed to focus log widget: {e}")
            self.notify("Could not focus logs widget", severity="warning")

    async def action_panic(self) -> None:
        """Trigger panic modal."""
        from gpt_trader.tui.widgets.panic import PanicModal

        def handle_panic(confirmed: bool) -> None:
            if confirmed:
                asyncio.create_task(self._execute_panic())

        self.push_screen(PanicModal(), handle_panic)

    async def _execute_panic(self) -> None:
        """Execute panic sequence."""
        try:
            self.notify("Executing FLATTEN & STOP...", severity="error", timeout=10)
            logger.critical("User initiated PANIC: flatten_and_stop sequence starting")
            messages = await self.bot.flatten_and_stop()
            for msg in messages:
                logger.info(f"Panic sequence message: {msg}")
                self.notify(
                    msg, severity="warning" if "Error" in msg else "information", timeout=10
                )
            logger.critical("PANIC sequence completed")
        except Exception as e:
            logger.critical(f"PANIC sequence failed: {e}", exc_info=True)
            self.notify(f"PANIC FAILED: {e}", severity="error", timeout=30)

    def on_bot_status_widget_toggle_bot_pressed(
        self, message: BotStatusWidget.ToggleBotPressed
    ) -> None:
        """Handle start/stop button press from BotStatusWidget."""
        asyncio.create_task(self.action_toggle_bot())

    def on_bot_status_widget_panic_pressed(self, message: BotStatusWidget.PanicPressed) -> None:
        """Handle panic button press from BotStatusWidget."""
        asyncio.create_task(self.action_panic())

    async def action_quit(self) -> None:
        """Quit the application."""
        try:
            logger.info("User initiated TUI shutdown")
            if self.bot.running and self._bot_task:
                logger.info("Stopping bot before TUI exit")

                # Cancel bot task
                self._bot_task.cancel()
                try:
                    await self._bot_task
                except asyncio.CancelledError:
                    logger.info("Bot task cancelled on quit")

                # Cleanup
                await self.bot.stop()
                self._bot_task = None
                logger.info("Bot stopped successfully")

            self.exit()
        except Exception as e:
            logger.error(f"Error during TUI shutdown: {e}", exc_info=True)
            # Force exit even if cleanup fails
            self.exit()

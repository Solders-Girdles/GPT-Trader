"""
Action Dispatcher Service for TUI.

Centralizes handling of user action bindings (keyboard shortcuts).
Extracted from TraderApp to reduce app.py complexity and improve testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp

logger = get_logger(__name__, component="tui")


class ActionDispatcher:
    """Handles user action bindings for the TUI.

    This service centralizes the implementation of keyboard shortcut actions,
    delegating to appropriate services and managers as needed.

    Attributes:
        app: Reference to the TraderApp instance.
    """

    def __init__(self, app: TraderApp) -> None:
        """Initialize the ActionDispatcher.

        Args:
            app: Reference to the TraderApp instance.
        """
        self.app = app

    async def toggle_bot(self) -> None:
        """Toggle bot running state.

        Delegates to BotLifecycleManager.
        """
        if self.app.lifecycle_manager:
            await self.app.lifecycle_manager.toggle_bot()

    async def show_config(self) -> None:
        """Show configuration modal.

        Delegates to ConfigService.
        """
        if self.app.bot and hasattr(self.app.bot, "config"):
            self.app.config_service.show_config_modal(self.app.bot.config)
        else:
            self.app.notify("No configuration available", severity="warning")

    async def focus_logs(self) -> None:
        """Focus the log widget on the current screen."""
        from gpt_trader.tui.screens import FullLogsScreen, MainScreen

        try:
            # Determine which screen we're on and query appropriate widget
            if isinstance(self.app.screen, MainScreen):
                log_widget = self.app.query_one("#dash-logs")
            elif isinstance(self.app.screen, FullLogsScreen):
                log_widget = self.app.query_one("#full-logs")
            else:
                # Other screens may not have log widgets
                self.app.notify("No log widget on this screen", severity="information")
                return

            log_widget.focus()
        except Exception as e:
            logger.warning(f"Failed to focus log widget: {e}")
            self.app.notify("Could not focus logs widget", severity="warning")

    async def show_full_logs(self) -> None:
        """Show full logs screen (expanded view)."""
        from gpt_trader.tui.screens import FullLogsScreen

        try:
            logger.debug("Opening full logs screen")
            self.app.push_screen(FullLogsScreen())
        except Exception as e:
            logger.error(f"Failed to show full logs screen: {e}", exc_info=True)
            self.app.notify(f"Error showing full logs: {e}", severity="error")

    async def show_system_details(self) -> None:
        """Show detailed system metrics screen."""
        from gpt_trader.tui.screens import SystemDetailsScreen

        try:
            logger.debug("Opening system details screen")
            self.app.push_screen(SystemDetailsScreen())
        except Exception as e:
            logger.error(f"Failed to show system details screen: {e}", exc_info=True)
            self.app.notify(f"Error showing system details: {e}", severity="error")

    async def show_mode_info(self) -> None:
        """Show detailed mode information modal."""
        from gpt_trader.tui.widgets import ModeInfoModal

        try:
            logger.debug(f"Opening mode info modal for mode: {self.app.data_source_mode}")
            self.app.push_screen(ModeInfoModal(self.app.data_source_mode))
        except Exception as e:
            logger.error(f"Failed to show mode info modal: {e}", exc_info=True)
            self.app.notify(f"Error showing mode info: {e}", severity="error")

    async def show_help(self) -> None:
        """Show comprehensive keyboard shortcut help screen."""
        try:
            from gpt_trader.tui.screens.help import HelpScreen

            logger.debug("Opening help screen")
            self.app.push_screen(HelpScreen())
        except Exception as e:
            logger.error(f"Failed to show help screen: {e}", exc_info=True)
            self.app.notify(f"Error showing help: {e}", severity="error")

    async def reconnect_data(self) -> None:
        """Attempt to reconnect data source."""
        try:
            if self.app.data_source_mode == "demo":
                self.app.notify("Demo mode doesn't require reconnection", severity="information")
                return

            logger.info("User initiated data source reconnection")

            # Trigger a status sync (delegated to UICoordinator)
            if self.app.ui_coordinator:
                self.app.ui_coordinator.sync_state_from_bot()

            # Check connection health immediately (no artificial delay)
            is_healthy = self.app.tui_state.check_connection_health()
            if is_healthy:
                self.app.notify(
                    "Data refreshed successfully", title="Connection", severity="information"
                )
            else:
                self.app.notify(
                    "Waiting for data update...",
                    title="Connection",
                    severity="warning",
                )
        except Exception as e:
            logger.error(f"Failed to reconnect: {e}", exc_info=True)
            self.app.notify(f"Error reconnecting: {e}", severity="error")

    async def panic(self) -> None:
        """Show panic confirmation modal for emergency stop.

        Opens PanicModal which requires typing 'FLATTEN' to confirm.
        On confirmation, stops bot and closes all positions immediately.
        """
        from gpt_trader.tui.widgets.panic import PanicModal

        def handle_panic_confirm(confirmed: bool) -> None:
            """Handle panic modal result."""
            if confirmed:
                logger.warning("User triggered PANIC - emergency stop initiated")
                self.app.notify(
                    "PANIC INITIATED - Stopping bot and flattening all positions",
                    title="Emergency Stop",
                    severity="error",
                    timeout=30,
                )
                # Delegate to lifecycle manager for safe shutdown
                try:
                    if self.app.lifecycle_manager:
                        self.app.lifecycle_manager.panic_stop()
                except Exception as e:
                    logger.error(f"Panic stop failed: {e}", exc_info=True)
                    self.app.notify(
                        f"PANIC STOP FAILED: {e}",
                        title="Critical Error",
                        severity="error",
                        timeout=60,
                    )
            else:
                logger.info("Panic cancelled by user")
                self.app.notify("Panic cancelled", severity="information")

        self.app.push_screen(PanicModal(), handle_panic_confirm)

    async def toggle_theme(self) -> None:
        """Toggle between dark and light themes.

        Delegates to ThemeService.
        """
        try:
            self.app.theme_service.toggle_theme()
        except Exception as e:
            logger.error(f"Failed to toggle theme: {e}", exc_info=True)
            self.app.notify(f"Theme switch failed: {e}", severity="error")

    async def show_alert_history(self) -> None:
        """Show alert history screen."""
        from gpt_trader.tui.screens import AlertHistoryScreen

        try:
            logger.debug("Opening alert history screen")
            self.app.push_screen(AlertHistoryScreen())
        except Exception as e:
            logger.error(f"Failed to show alert history screen: {e}", exc_info=True)
            self.app.notify(f"Error showing alerts: {e}", severity="error")

    async def force_refresh(self) -> None:
        """Force a full refresh of bot state and UI.

        Useful when connection might be stale or data seems outdated.
        """
        try:
            logger.info("User initiated force refresh")
            self.app.notify("Refreshing bot state...", severity="information")

            # Sync state from bot
            if self.app.ui_coordinator:
                self.app.ui_coordinator.sync_state_from_bot()

            # Update the main screen
            if self.app.ui_coordinator:
                self.app.ui_coordinator.update_main_screen()

            # Reset alert cooldowns to allow re-triggering if conditions still exist
            if hasattr(self.app, "alert_manager"):
                self.app.alert_manager.reset_cooldowns()
                # Re-check alerts with fresh state
                self.app.alert_manager.check_alerts(self.app.tui_state)

            self.app.notify("Refresh complete", severity="information")
            logger.info("Force refresh completed")
        except Exception as e:
            logger.error(f"Force refresh failed: {e}", exc_info=True)
            self.app.notify(f"Refresh failed: {e}", severity="error")

    async def quit_app(self) -> None:
        """Quit the application gracefully."""
        try:
            logger.info("User initiated TUI shutdown")
            if self.app.bot and self.app.bot.running:
                logger.info("Stopping bot before TUI exit")

                # Stop bot via lifecycle manager
                if self.app.lifecycle_manager:
                    await self.app.lifecycle_manager.stop_bot()
                else:
                    # Fallback if manager doesn't exist
                    await self.app.bot.stop()

                logger.info("Bot stopped successfully")

            self.app.exit()
        except Exception as e:
            logger.error(f"Error during TUI shutdown: {e}", exc_info=True)
            # Force exit even if cleanup fails
            self.app.exit()

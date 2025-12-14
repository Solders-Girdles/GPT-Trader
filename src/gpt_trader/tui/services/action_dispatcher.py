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
        """Start/refresh data feed.

        Context-aware behavior:
        - If data not yet available: "Starting data feed..."
        - If data available: "Refreshing data..."
        - In demo mode: "Demo mode uses simulated data"
        """
        try:
            if self.app.data_source_mode == "demo":
                self.app.notify("Demo mode uses simulated data", severity="information")
                return

            # Set data_fetching flag for UI feedback
            self.app.tui_state.data_fetching = True

            # Context-aware notification
            if not self.app.tui_state.data_available:
                self.app.notify("Starting data feed...", severity="information")
                logger.info("User initiated data feed start")
            else:
                self.app.notify("Refreshing data...", severity="information")
                logger.info("User initiated data refresh")

            # Even when STOPPED, fetch a fresh account/market snapshot so the UI
            # shows balances and baseline market data without running the bot.
            try:
                request_bootstrap = getattr(self.app, "request_bootstrap_snapshot", None)
                if callable(request_bootstrap):
                    request_bootstrap(force=True)
            except Exception:
                pass

            # Trigger a status sync (delegated to UICoordinator)
            if self.app.ui_coordinator:
                self.app.ui_coordinator.sync_state_from_bot()

            # Clear fetching flag
            self.app.tui_state.data_fetching = False

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
            self.app.tui_state.data_fetching = False
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

    # =========================================================================
    # Watchlist Actions
    # =========================================================================

    async def edit_watchlist(self) -> None:
        """Open the watchlist editor modal."""
        from gpt_trader.tui.screens.watchlist_screen import WatchlistScreen

        try:
            logger.debug("Opening watchlist editor")
            self.app.push_screen(WatchlistScreen())
        except Exception as e:
            logger.error(f"Failed to open watchlist editor: {e}", exc_info=True)
            self.app.notify(f"Error opening watchlist: {e}", severity="error")

    async def clear_watchlist(self) -> None:
        """Clear all symbols from the watchlist."""
        from gpt_trader.tui.services.preferences_service import get_preferences_service

        try:
            prefs = get_preferences_service()
            prefs.clear_watchlist()
            self.app.notify("Watchlist cleared", severity="information")
            logger.info("User cleared watchlist")
        except Exception as e:
            logger.error(f"Failed to clear watchlist: {e}", exc_info=True)
            self.app.notify(f"Error clearing watchlist: {e}", severity="error")

    # =========================================================================
    # Operator/Diagnostics Actions
    # =========================================================================

    async def diagnose_connection(self) -> None:
        """Run connection diagnostics and display results."""
        try:
            logger.info("Running connection diagnostics")
            self.app.notify("Running diagnostics...", severity="information")

            results = []

            # Check data source mode
            mode = self.app.data_source_mode
            results.append(f"Mode: {mode}")

            # Check bot status
            if self.app.bot:
                results.append(f"Bot running: {self.app.bot.running}")
            else:
                results.append("Bot: Not initialized")

            # Check TUI state health
            state = self.app.tui_state
            results.append(f"Data available: {state.data_available}")
            results.append(f"Connection healthy: {state.check_connection_health()}")
            results.append(f"Degraded mode: {state.degraded_mode}")

            # Check resilience metrics
            resilience = state.resilience_data
            results.append(f"API latency: {resilience.api_latency_ms:.0f}ms")
            results.append(f"Error rate: {resilience.error_rate_pct:.1f}%")
            results.append(f"Circuit breakers open: {resilience.circuit_breaker_open_count}")

            # Display results
            diagnostics_text = "\n".join(results)
            self.app.notify(
                diagnostics_text,
                title="Connection Diagnostics",
                severity="information",
                timeout=15,
            )
            logger.info(f"Diagnostics complete: {results}")
        except Exception as e:
            logger.error(f"Diagnostics failed: {e}", exc_info=True)
            self.app.notify(f"Diagnostics failed: {e}", severity="error")

    async def export_logs(self) -> None:
        """Export current logs to a timestamped file."""
        import time
        from pathlib import Path

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            export_path = Path(f"logs/tui_export_{timestamp}.log")
            export_path.parent.mkdir(parents=True, exist_ok=True)

            # Get logs from log manager
            if hasattr(self.app, "log_manager"):
                logs = self.app.log_manager.get_recent_logs(limit=1000)
                with open(export_path, "w") as f:
                    for log in logs:
                        f.write(f"{log}\n")

                self.app.notify(
                    f"Logs exported to {export_path}",
                    title="Export Complete",
                    severity="information",
                )
                logger.info(f"Exported {len(logs)} log entries to {export_path}")
            else:
                self.app.notify("Log manager not available", severity="warning")
        except Exception as e:
            logger.error(f"Log export failed: {e}", exc_info=True)
            self.app.notify(f"Export failed: {e}", severity="error")

    async def export_state(self) -> None:
        """Export current TuiState as JSON snapshot."""
        import json
        import time
        from pathlib import Path

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            export_path = Path(f"logs/state_snapshot_{timestamp}.json")
            export_path.parent.mkdir(parents=True, exist_ok=True)

            state = self.app.tui_state

            # Build exportable state dict
            snapshot = {
                "timestamp": time.time(),
                "mode": self.app.data_source_mode,
                "running": state.running,
                "data_available": state.data_available,
                "degraded_mode": state.degraded_mode,
                "market_data": {
                    "symbols": list(state.market_data.prices.keys()),
                    "last_update": state.market_data.last_update,
                },
                "position_data": {
                    "position_count": len(state.position_data.positions),
                    "equity": str(state.position_data.equity),
                    "total_pnl": str(state.position_data.total_unrealized_pnl),
                },
                "account_data": {
                    "volume_30d": str(state.account_data.volume_30d),
                    "fees_30d": str(state.account_data.fees_30d),
                    "fee_tier": state.account_data.fee_tier,
                    "balance_count": len(state.account_data.balances),
                },
                "resilience_data": {
                    "api_latency_ms": state.resilience_data.api_latency_ms,
                    "error_rate_pct": state.resilience_data.error_rate_pct,
                    "circuit_breaker_open_count": state.resilience_data.circuit_breaker_open_count,
                },
            }

            with open(export_path, "w") as f:
                json.dump(snapshot, f, indent=2)

            self.app.notify(
                f"State exported to {export_path}",
                title="Export Complete",
                severity="information",
            )
            logger.info(f"Exported state snapshot to {export_path}")
        except Exception as e:
            logger.error(f"State export failed: {e}", exc_info=True)
            self.app.notify(f"Export failed: {e}", severity="error")

    async def reconnect_websocket(self) -> None:
        """Force WebSocket reconnection."""
        try:
            logger.info("User initiated WebSocket reconnection")
            self.app.notify("Reconnecting WebSocket...", severity="information")

            # Check if bot has a client with websocket
            if self.app.bot and hasattr(self.app.bot, "client"):
                client = self.app.bot.client
                if hasattr(client, "reconnect_websocket"):
                    await client.reconnect_websocket()
                    self.app.notify(
                        "WebSocket reconnection initiated",
                        severity="information",
                    )
                elif hasattr(client, "ws") and hasattr(client.ws, "reconnect"):
                    await client.ws.reconnect()
                    self.app.notify(
                        "WebSocket reconnection initiated",
                        severity="information",
                    )
                else:
                    self.app.notify(
                        "WebSocket reconnect not available for this client",
                        severity="warning",
                    )
            else:
                self.app.notify("No active client connection", severity="warning")
        except Exception as e:
            logger.error(f"WebSocket reconnection failed: {e}", exc_info=True)
            self.app.notify(f"Reconnection failed: {e}", severity="error")

    async def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers to closed state."""
        try:
            logger.info("User initiated circuit breaker reset")

            reset_count = 0

            # Check if bot has circuit breakers
            if self.app.bot and hasattr(self.app.bot, "client"):
                client = self.app.bot.client

                # Try various circuit breaker locations
                if hasattr(client, "circuit_breaker"):
                    client.circuit_breaker.reset()
                    reset_count += 1

                if hasattr(client, "_circuit_breakers"):
                    for cb in client._circuit_breakers.values():
                        cb.reset()
                        reset_count += 1

            # Also check resilience manager if exists
            if hasattr(self.app.bot, "resilience_manager"):
                rm = self.app.bot.resilience_manager
                if hasattr(rm, "reset_all"):
                    rm.reset_all()
                    reset_count += 1

            if reset_count > 0:
                self.app.notify(
                    f"Reset {reset_count} circuit breaker(s)",
                    severity="information",
                )
                logger.info(f"Reset {reset_count} circuit breakers")
            else:
                self.app.notify(
                    "No circuit breakers found to reset",
                    severity="warning",
                )
        except Exception as e:
            logger.error(f"Circuit breaker reset failed: {e}", exc_info=True)
            self.app.notify(f"Reset failed: {e}", severity="error")

"""
Action Dispatcher Service for TUI.

Centralizes handling of user action bindings (keyboard shortcuts).
Extracted from TraderApp to reduce app.py complexity and improve testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpt_trader.tui.notification_helpers import (
    notify_action,
    notify_error,
    notify_success,
    notify_warning,
)
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
            notify_warning(self.app, "No configuration available")

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
                notify_action(self.app, "No log widget on this screen")
                return

            log_widget.focus()
        except Exception as e:
            logger.warning(f"Failed to focus log widget: {e}")
            notify_warning(self.app, "Could not focus logs widget")

    async def show_full_logs(self) -> None:
        """Show full logs screen (expanded view)."""
        from gpt_trader.tui.screens import FullLogsScreen

        try:
            logger.debug("Opening full logs screen")
            self.app.push_screen(FullLogsScreen())
        except Exception as e:
            logger.error(f"Failed to show full logs screen: {e}", exc_info=True)
            notify_error(self.app, f"Error showing full logs: {e}")

    async def show_system_details(self) -> None:
        """Show detailed system metrics screen."""
        from gpt_trader.tui.screens import SystemDetailsScreen

        try:
            logger.debug("Opening system details screen")
            self.app.push_screen(SystemDetailsScreen())
        except Exception as e:
            logger.error(f"Failed to show system details screen: {e}", exc_info=True)
            notify_error(self.app, f"Error showing system details: {e}")

    async def show_mode_info(self) -> None:
        """Show detailed mode information modal."""
        from gpt_trader.tui.widgets import ModeInfoModal

        try:
            logger.debug(f"Opening mode info modal for mode: {self.app.data_source_mode}")
            self.app.push_screen(ModeInfoModal(self.app.data_source_mode))
        except Exception as e:
            logger.error(f"Failed to show mode info modal: {e}", exc_info=True)
            notify_error(self.app, f"Error showing mode info: {e}")

    async def show_help(self) -> None:
        """Show comprehensive keyboard shortcut help screen."""
        try:
            from gpt_trader.tui.screens.help import HelpScreen

            logger.debug("Opening help screen")
            self.app.push_screen(HelpScreen())
        except Exception as e:
            logger.error(f"Failed to show help screen: {e}", exc_info=True)
            notify_error(self.app, f"Error showing help: {e}")

    async def reconnect_data(self) -> None:
        """Start/refresh data feed.

        Context-aware behavior:
        - If data not yet available: "Starting data feed..."
        - If data available: "Refreshing data..."
        - In demo mode: "Demo mode uses simulated data"
        """
        try:
            if self.app.data_source_mode == "demo":
                notify_action(self.app, "Demo mode uses simulated data")
                return

            # Set data_fetching flag for UI feedback
            self.app.tui_state.data_fetching = True

            # Context-aware notification
            if not self.app.tui_state.data_available:
                notify_action(self.app, "Starting data feed...")
                logger.info("User initiated data feed start")
            else:
                notify_action(self.app, "Refreshing data...")
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
                notify_success(self.app, "Data refreshed successfully", title="Connection")
            else:
                notify_warning(self.app, "Waiting for data update...", title="Connection")
        except Exception as e:
            self.app.tui_state.data_fetching = False
            logger.error(f"Failed to reconnect: {e}", exc_info=True)
            notify_error(self.app, f"Error reconnecting: {e}")

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
                notify_error(
                    self.app,
                    "PANIC INITIATED - Stopping bot and flattening all positions",
                    title="Emergency Stop",
                    timeout=30,
                )
                # Delegate to lifecycle manager for safe shutdown
                try:
                    if self.app.lifecycle_manager:
                        self.app.lifecycle_manager.panic_stop()
                except Exception as e:
                    logger.error(f"Panic stop failed: {e}", exc_info=True)
                    notify_error(
                        self.app,
                        f"PANIC STOP FAILED: {e}",
                        title="Critical Error",
                        timeout=60,
                    )
            else:
                logger.info("Panic cancelled by user")
                notify_action(self.app, "Panic cancelled")

        self.app.push_screen(PanicModal(), handle_panic_confirm)

    async def toggle_theme(self) -> None:
        """Toggle between dark and light themes.

        Delegates to ThemeService.
        """
        try:
            self.app.theme_service.toggle_theme()
        except Exception as e:
            logger.error(f"Failed to toggle theme: {e}", exc_info=True)
            notify_error(self.app, f"Theme switch failed: {e}")

    async def show_alert_history(self) -> None:
        """Show alert history screen."""
        from gpt_trader.tui.screens import AlertHistoryScreen

        try:
            logger.debug("Opening alert history screen")
            self.app.push_screen(AlertHistoryScreen())
        except Exception as e:
            logger.error(f"Failed to show alert history screen: {e}", exc_info=True)
            notify_error(self.app, f"Error showing alerts: {e}")

    async def force_refresh(self) -> None:
        """Force a full refresh of bot state and UI.

        Useful when connection might be stale or data seems outdated.
        """
        try:
            logger.info("User initiated force refresh")
            notify_action(self.app, "Refreshing bot state...")

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

            notify_success(self.app, "Refresh complete")
            logger.info("Force refresh completed")
        except Exception as e:
            logger.error(f"Force refresh failed: {e}", exc_info=True)
            notify_error(self.app, f"Refresh failed: {e}")

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
            notify_error(self.app, f"Error opening watchlist: {e}")

    async def clear_watchlist(self) -> None:
        """Clear all symbols from the watchlist."""
        from gpt_trader.tui.services.preferences_service import get_preferences_service

        try:
            prefs = get_preferences_service()
            prefs.clear_watchlist()
            notify_success(self.app, "Watchlist cleared")
            logger.info("User cleared watchlist")
        except Exception as e:
            logger.error(f"Failed to clear watchlist: {e}", exc_info=True)
            notify_error(self.app, f"Error clearing watchlist: {e}")

    # =========================================================================
    # Operator/Diagnostics Actions
    # =========================================================================

    async def diagnose_connection(self) -> None:
        """Run connection diagnostics and display results."""
        try:
            logger.info("Running connection diagnostics")
            notify_action(self.app, "Running diagnostics...")

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
            notify_action(self.app, diagnostics_text, title="Connection Diagnostics", timeout=15)
            logger.info(f"Diagnostics complete: {results}")
        except Exception as e:
            logger.error(f"Diagnostics failed: {e}", exc_info=True)
            notify_error(self.app, f"Diagnostics failed: {e}")

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

                notify_success(self.app, f"Logs exported to {export_path}", title="Export Complete")
                logger.info(f"Exported {len(logs)} log entries to {export_path}")
            else:
                notify_warning(self.app, "Log manager not available")
        except Exception as e:
            logger.error(f"Log export failed: {e}", exc_info=True)
            notify_error(self.app, f"Export failed: {e}")

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

            notify_success(self.app, f"State exported to {export_path}", title="Export Complete")
            logger.info(f"Exported state snapshot to {export_path}")
        except Exception as e:
            logger.error(f"State export failed: {e}", exc_info=True)
            notify_error(self.app, f"Export failed: {e}")

    async def reconnect_websocket(self) -> None:
        """Force WebSocket reconnection."""
        try:
            logger.info("User initiated WebSocket reconnection")
            notify_action(self.app, "Reconnecting WebSocket...")

            # Check if bot has a client with websocket
            if self.app.bot and hasattr(self.app.bot, "client"):
                client = self.app.bot.client
                if hasattr(client, "reconnect_websocket"):
                    await client.reconnect_websocket()
                    notify_success(self.app, "WebSocket reconnection initiated")
                elif hasattr(client, "ws") and hasattr(client.ws, "reconnect"):
                    await client.ws.reconnect()
                    notify_success(self.app, "WebSocket reconnection initiated")
                else:
                    notify_warning(self.app, "WebSocket reconnect not available for this client")
            else:
                notify_warning(self.app, "No active client connection")
        except Exception as e:
            logger.error(f"WebSocket reconnection failed: {e}", exc_info=True)
            notify_error(self.app, f"Reconnection failed: {e}")

    async def reset_daily_risk(self) -> None:
        """Reset daily risk tracking (P&L, limits).

        Calls reset_daily_tracking() on the execution engine if available.
        Useful at the start of a new trading day.
        """
        try:
            logger.info("User initiated daily risk reset")

            # Check if bot has execution engine with reset method
            if self.app.bot and hasattr(self.app.bot, "execution_engine"):
                engine = self.app.bot.execution_engine
                if hasattr(engine, "reset_daily_tracking"):
                    engine.reset_daily_tracking()
                    notify_success(self.app, "Daily risk tracking reset", title="Risk Reset")
                    logger.info("Daily risk tracking reset completed")
                    return

            # Fallback: check for risk manager directly
            if self.app.bot and hasattr(self.app.bot, "risk_manager"):
                risk_manager = self.app.bot.risk_manager
                if hasattr(risk_manager, "reset_daily_tracking"):
                    risk_manager.reset_daily_tracking()
                    notify_success(self.app, "Daily risk tracking reset", title="Risk Reset")
                    logger.info("Daily risk tracking reset completed via risk manager")
                    return

            notify_warning(self.app, "Reset not available - no active bot engine")
        except Exception as e:
            logger.error(f"Daily risk reset failed: {e}", exc_info=True)
            notify_error(self.app, f"Risk reset failed: {e}")

    async def enable_reduce_only(self) -> None:
        """Enable reduce-only mode to prevent new position entries.

        Sets reduce_only_mode on the risk manager with reason "operator_reduce_only".
        Only allows closing/reducing existing positions until manually cleared.
        """
        try:
            logger.info("User initiated reduce-only mode")

            # Check if bot has risk manager
            if self.app.bot and hasattr(self.app.bot, "risk_manager"):
                risk_manager = self.app.bot.risk_manager

                # Check if already in reduce-only mode
                if hasattr(risk_manager, "reduce_only_mode") and risk_manager.reduce_only_mode:
                    notify_warning(self.app, "Already in reduce-only mode", title="Risk")
                    return

                # Enable reduce-only mode
                if hasattr(risk_manager, "set_reduce_only_mode"):
                    risk_manager.set_reduce_only_mode(True, reason="operator_reduce_only")
                    notify_success(
                        self.app,
                        "Reduce-only mode enabled - no new entries",
                        title="Risk",
                    )
                    logger.info("Reduce-only mode enabled by operator")
                    return

            notify_warning(self.app, "No risk manager available")
        except Exception as e:
            logger.error(f"Enable reduce-only failed: {e}", exc_info=True)
            notify_error(self.app, f"Failed to enable reduce-only: {e}")

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
                notify_success(self.app, f"Reset {reset_count} circuit breaker(s)")
                logger.info(f"Reset {reset_count} circuit breakers")
            else:
                notify_warning(self.app, "No circuit breakers found to reset")
        except Exception as e:
            logger.error(f"Circuit breaker reset failed: {e}", exc_info=True)
            notify_error(self.app, f"Reset failed: {e}")

"""
UI Coordinator for TUI.

Handles UI update orchestration from bot status updates via observer callbacks.
Provides heartbeat animation loop. Extracted from TraderApp to reduce complexity.

Supports optional update throttling for high-frequency updates.
Includes performance instrumentation for monitoring update cycle timing.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from gpt_trader.monitoring.status_reporter import BotStatus
from gpt_trader.tui.notification_helpers import notify_error, notify_success
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp
    from gpt_trader.tui.services.update_throttler import UpdateThrottler

logger = get_logger(__name__, component="tui")


class UICoordinator:
    """
    Coordinates UI updates from bot status changes.

    Data updates are driven by observer callbacks (apply_observer_update).
    Heartbeat loop provides visual animation (no data polling).

    Supports optional update throttling to batch high-frequency updates.
    """

    def __init__(
        self,
        app: TraderApp,
        throttler: UpdateThrottler | None = None,
    ):
        """
        Initialize UICoordinator.

        Args:
            app: Reference to the TraderApp instance
            throttler: Optional UpdateThrottler for batching updates
        """
        self.app = app
        self._update_task: asyncio.Task | None = None  # Heartbeat task
        self._throttler = throttler
        self._use_throttling = throttler is not None

        # Set up throttler callback if provided
        if self._throttler:
            self._throttler.set_flush_callback(self._apply_throttled_updates)
            logger.debug("UICoordinator configured with update throttling")

    def apply_observer_update(self, status: BotStatus) -> None:
        """
        Apply typed status update to state and UI.

        Called from observer callback on bot status changes. Protected with
        error boundaries to prevent cascade failures.

        When throttling is enabled, updates are queued and batched.

        Args:
            status: Typed BotStatus snapshot from StatusReporter
        """
        # If throttling is enabled, queue the update instead of applying immediately
        if self._use_throttling and self._throttler:
            self._throttler.queue_full_status(status)
            logger.debug("Status update queued for throttled processing")
            return

        # Apply immediately (no throttling)
        self._apply_status_update(status)

    def _apply_status_update(self, status: BotStatus) -> None:
        """
        Internal method to apply a status update.

        Used both for immediate updates and by the throttler callback.
        Includes performance instrumentation for monitoring.

        Args:
            status: Typed BotStatus snapshot from StatusReporter
        """
        # Import here to avoid circular imports
        from gpt_trader.tui.services.performance_service import (
            FrameMetrics,
            get_tui_performance_service,
        )

        perf = get_tui_performance_service()
        frame_start = time.time()
        state_duration = 0.0
        render_duration = 0.0

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
            state_start = time.time()
            try:
                self.app.tui_state.running = self.app.bot.running
                self.app.tui_state.update_from_bot_status(status, runtime_state)

                # Mark data as available and update fetch timestamp
                self.app.tui_state.data_available = True
                self.app.tui_state.last_data_fetch = time.time()

                # Log successful update (defensive against test stubs / alt payload shapes)
                positions_count = 0
                if status.positions:
                    positions = getattr(status.positions, "positions", None)
                    if isinstance(positions, dict):
                        positions_count = len(positions)
                    elif hasattr(positions, "__dict__"):
                        positions_count = len(getattr(positions, "__dict__", {}) or {})
                    else:
                        try:
                            positions_count = len(positions) if positions is not None else 0
                        except TypeError:
                            positions_count = 0

                market_count = 0
                if status.market:
                    last_prices = getattr(status.market, "last_prices", None)
                    if isinstance(last_prices, dict):
                        market_count = len(last_prices)
                    elif hasattr(last_prices, "__dict__"):
                        market_count = len(getattr(last_prices, "__dict__", {}) or {})
                    else:
                        try:
                            market_count = len(last_prices) if last_prices is not None else 0
                        except TypeError:
                            market_count = 0

                logger.debug(
                    "State updated successfully: positions=%s, market_symbols=%s",
                    positions_count,
                    market_count,
                )
            except Exception as e:
                logger.error(f"Failed to update TuiState from bot status: {e}", exc_info=True)
                notify_error(self.app, "Critical: State update failed. UI may show stale data.", title="State Update Error")
                # Don't proceed to UI update if state update failed
                return
            finally:
                state_duration = time.time() - state_start

            # Update UI (already has some error handling in update_main_screen)
            render_start = time.time()
            self.update_main_screen()
            render_duration = time.time() - render_start

            # Check alert rules against new state
            if hasattr(self.app, "alert_manager"):
                try:
                    self.app.alert_manager.check_alerts(self.app.tui_state)
                except Exception as alert_error:
                    logger.debug(f"Error checking alerts: {alert_error}")

        except Exception as e:
            logger.error(f"Critical error in apply_observer_update: {e}", exc_info=True)
            # Last resort notification
            try:
                notify_error(self.app, "Critical UI update failure. Consider restarting TUI.", title="Critical Error", timeout=30)
            except Exception:
                # Even notification failed - log and continue
                logger.critical("Cannot notify user of critical error - TUI may be unresponsive")
        finally:
            # Record frame metrics for performance monitoring
            frame_end = time.time()
            metrics = FrameMetrics(
                timestamp=frame_end,
                total_duration=frame_end - frame_start,
                state_update_duration=state_duration,
                widget_render_duration=render_duration,
            )
            perf.record_frame(metrics)

    def _apply_throttled_updates(self, updates: dict[str, Any]) -> None:
        """
        Callback from throttler to apply batched updates.

        Args:
            updates: Dict of component names to update data
        """
        # Check for full_status update (most common case)
        if "full_status" in updates:
            status = updates["full_status"]
            if isinstance(status, BotStatus):
                self._apply_status_update(status)
                logger.debug("Applied throttled full status update")
            return

        # Handle individual component updates if needed in future
        logger.debug(f"Received {len(updates)} throttled component updates")

    def sync_state_from_bot(self) -> None:
        """
        Manually sync state from bot (for reconnect action).

        Fetches current status from StatusReporter and updates UI.
        Handles NullStatusReporter gracefully in degraded mode.
        """
        self.app.tui_state.running = self.app.bot.running

        # Access runtime state safely
        runtime_state = None
        if hasattr(self.app.bot.engine.context, "runtime_state"):
            runtime_state = self.app.bot.engine.context.runtime_state

        # Access StatusReporter for typed data
        if hasattr(self.app.bot.engine, "status_reporter"):
            reporter = self.app.bot.engine.status_reporter

            # Check if this is a NullStatusReporter (degraded mode)
            if getattr(reporter, "is_null_reporter", False):
                logger.debug("NullStatusReporter detected, skipping sync (degraded mode)")
                self.app.tui_state.connection_healthy = False
                return

            status = reporter.get_status()  # Returns BotStatus
            logger.debug(
                f"Fetched status from StatusReporter: bot_id={status.bot_id}, "
                f"timestamp={status.timestamp_iso}"
            )
            self.app.tui_state.update_from_bot_status(status, runtime_state)

            # Mark data as available and update fetch timestamp
            self.app.tui_state.data_available = True
            self.app.tui_state.last_data_fetch = time.time()
        else:
            logger.debug("No StatusReporter available, skipping status update")
            self.app.tui_state.connection_healthy = False

    def update_main_screen(self) -> None:
        """Update the main screen UI with current state."""
        try:
            from gpt_trader.tui.screens import MainScreen, SystemDetailsScreen

            main_screen = self.app.query_one(MainScreen)
            main_screen.update_ui(self.app.tui_state)

            # Propagate state to SystemDetailsScreen if it's showing
            try:
                system_details = self.app.query_one(SystemDetailsScreen)
                system_details.state = self.app.tui_state
            except Exception:
                # SystemDetailsScreen not mounted - that's fine
                pass

            # Broadcast state on every UI update so StateRegistry observers update
            # even when the active screen doesn't change (TuiState is mutated in-place).
            try:
                if hasattr(self.app, "state_registry"):
                    self.app.state_registry.broadcast(self.app.tui_state)  # type: ignore[attr-defined]
            except Exception:
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
        logger.debug("Starting UI heartbeat loop")
        self._update_task = asyncio.create_task(self._heartbeat_loop())

    async def stop_update_loop(self) -> None:
        """Stop the periodic heartbeat loop."""
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                logger.debug("Heartbeat loop cancelled successfully")
            self._update_task = None
            logger.debug("UI heartbeat loop stopped")

    async def _heartbeat_loop(self) -> None:
        """
        Periodic heartbeat loop (runs every 2 seconds).

        Only pulses the heartbeat animation when bot is running AND
        the status widget is visible to minimize unnecessary work.
        Data updates are handled by the observer callback (no polling needed).

        Also periodically checks for StatusReporter reconnection in degraded mode.
        Collects resilience metrics every ~6 seconds for System tile display.
        """
        loop_count = 0
        reconnect_check_interval = 30  # Check every 60 seconds (30 loops * 2s)
        resilience_check_interval = 3  # Every 6 seconds (3 loops * 2s)
        spread_check_interval = 5  # Every 10 seconds (5 loops * 2s)
        while True:
            try:
                loop_count += 1
                if loop_count % 30 == 0:  # Log every minute
                    logger.debug(f"Heartbeat loop iteration {loop_count}")

                # Only pulse heartbeat when bot is running AND widget is visible
                should_pulse = (
                    self.app.bot is not None
                    and self.app.bot.running
                    and self._is_status_widget_visible()
                )

                if should_pulse:
                    self.app._pulse_heartbeat()

                # Update connection health even when no new status snapshots arrive.
                # Broadcast only on transitions to avoid unnecessary re-renders.
                try:
                    before = self.app.tui_state.connection_healthy
                    after = self.app.tui_state.check_connection_health()
                    if after != before and hasattr(self.app, "state_registry"):
                        self.app.state_registry.broadcast(self.app.tui_state)  # type: ignore[attr-defined]
                except Exception:
                    pass

                # Check for StatusReporter reconnection in degraded mode
                if loop_count % reconnect_check_interval == 0:
                    await self._check_status_reporter_reconnection()

                # Collect resilience metrics periodically for System tile
                if loop_count % resilience_check_interval == 0:
                    self._collect_resilience_metrics()

                # Collect spread data periodically for Market tile (rate-limited)
                if loop_count % spread_check_interval == 0:
                    asyncio.create_task(self._collect_spread_data())

            except Exception as e:
                logger.debug(f"Heartbeat pulse error: {e}")

            await asyncio.sleep(2)  # 2 second interval is sufficient for visibility

    def _is_status_widget_visible(self) -> bool:
        """Check if the status widget is currently visible.

        Returns:
            True if the status widget is mounted and displayed.
        """
        try:
            from gpt_trader.tui.widgets import BotStatusWidget

            status_widget = self.app.query_one(BotStatusWidget)
            return status_widget.is_mounted and status_widget.display
        except Exception:
            # Widget not found or error - assume not visible
            return False

    async def _check_status_reporter_reconnection(self) -> None:
        """Check if a real StatusReporter has become available in degraded mode.

        Called periodically from the heartbeat loop to detect when the
        StatusReporter becomes available after starting in degraded mode.
        If a real reporter is detected, exits degraded mode and connects
        the observer.
        """
        # Only check if currently in degraded mode
        if not self.app.tui_state.degraded_mode:
            return

        if not self.app.bot or not hasattr(self.app.bot, "engine"):
            return

        reporter = getattr(self.app.bot.engine, "status_reporter", None)
        if reporter is None:
            return

        # Check if we now have a real reporter (not NullStatusReporter)
        if getattr(reporter, "is_null_reporter", False):
            return  # Still using NullStatusReporter

        # StatusReporter is now available - exit degraded mode
        logger.info("StatusReporter became available - exiting degraded mode")
        self.app.tui_state.degraded_mode = False
        self.app.tui_state.degraded_reason = ""
        self.app.tui_state.connection_healthy = True

        # Connect observer
        reporter.add_observer(self.app._on_status_update)

        # Notify user
        notify_success(self.app, "Data connection restored", title="Connected")

        # Sync state immediately
        self.sync_state_from_bot()
        logger.info("Reconnected to StatusReporter and synced state")

    def _collect_resilience_metrics(self) -> None:
        """Collect resilience metrics from CoinbaseClient if available.

        Navigates through the bot -> engine -> broker -> client chain
        to access the client's get_resilience_status() method.
        """
        try:
            if not self.app.bot:
                return

            # Navigate to client through broker
            engine = getattr(self.app.bot, "engine", None)
            if not engine:
                return
            context = getattr(engine, "context", None)
            if not context:
                return
            broker = getattr(context, "broker", None)
            if not broker:
                return
            client = getattr(broker, "_client", None)
            if not client:
                return

            if hasattr(client, "get_resilience_status"):
                status = client.get_resilience_status()
                self.app.tui_state.update_resilience_data(status)
                logger.debug("Collected resilience metrics from client")
        except Exception as e:
            logger.debug("Failed to collect resilience metrics: %s", e)

    def _get_client(self) -> Any | None:
        """Get CoinbaseClient from bot -> engine -> broker -> client chain.

        Returns:
            CoinbaseClient instance or None if not available.
        """
        try:
            if not self.app.bot:
                return None
            engine = getattr(self.app.bot, "engine", None)
            if not engine:
                return None
            context = getattr(engine, "context", None)
            if not context:
                return None
            broker = getattr(context, "broker", None)
            if not broker:
                return None
            return getattr(broker, "_client", None)
        except Exception:
            return None

    async def _collect_spread_data(self) -> None:
        """Collect spread data from order books for watched symbols.

        Fetches best bid/ask for each symbol and calculates spread percentage.
        Limited to 5 symbols to avoid rate limiting.
        """
        from decimal import Decimal

        try:
            client = self._get_client()
            if not client:
                return

            # Get symbols from current market data
            symbols = list(self.app.tui_state.market_data.prices.keys())[:5]
            if not symbols:
                return

            spreads: dict[str, Decimal] = {}
            for symbol in symbols:
                try:
                    # Get best bid/ask from order book (level 1)
                    if hasattr(client, "get_product_book"):
                        book = await client.get_product_book(symbol, limit=1)
                        if book and book.get("bids") and book.get("asks"):
                            best_bid = Decimal(str(book["bids"][0]["price"]))
                            best_ask = Decimal(str(book["asks"][0]["price"]))
                            if best_bid > 0:
                                spread_pct = ((best_ask - best_bid) / best_bid) * 100
                                spreads[symbol] = spread_pct
                except Exception:
                    pass  # Skip symbol on error

            # Update market data with spreads
            if spreads:
                # Preserve existing spreads not in current batch
                current_spreads = dict(self.app.tui_state.market_data.spreads)
                current_spreads.update(spreads)
                self.app.tui_state.market_data.spreads = current_spreads
                logger.debug("Collected spread data for %d symbols", len(spreads))
        except Exception as e:
            logger.debug("Failed to collect spread data: %s", e)

    def cleanup(self) -> None:
        """Clean up heartbeat tasks and throttler on manager destruction."""
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            logger.debug("UICoordinator cleaned up heartbeat task")

        if self._throttler:
            self._throttler.cancel_pending()
            logger.debug("UICoordinator cleaned up throttler")

"""Lifecycle mixin for TraderApp.

Contains methods related to app mount/unmount lifecycle,
initialization, and cleanup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual import events

from gpt_trader.tui.managers import BotLifecycleManager, UICoordinator
from gpt_trader.tui.notification_helpers import notify_action, notify_error, notify_warning
from gpt_trader.tui.responsive_state import ResponsiveState
from gpt_trader.tui.screens import MainScreen
from gpt_trader.tui.services import UpdateThrottler
from gpt_trader.tui.services.worker_service import WorkerService
from gpt_trader.tui.widgets import LiveWarningModal
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp

logger = get_logger(__name__, component="tui")


class TraderAppLifecycleMixin:
    """Mixin providing lifecycle methods for TraderApp.

    Methods:
    - on_mount: Called when app starts
    - _initialize_with_bot: Initialize with bot instance
    - on_unmount: Called when app stops
    - _cleanup_bot_resources: Clean up bot resources
    - on_resize: Handle terminal resize
    - watch_responsive_state: Propagate responsive state changes
    """

    # Type hints for attributes from TraderApp
    if TYPE_CHECKING:
        bot: Any
        mode_service: Any
        responsive_manager: Any
        lifecycle_manager: BotLifecycleManager | None
        ui_coordinator: UICoordinator | None
        worker_service: WorkerService | None
        data_source_mode: str
        tui_state: Any
        _initial_mode: str | None
        _demo_scenario: str
        responsive_state: ResponsiveState
        terminal_width: int
        size: Any

        def _detect_bot_mode(self) -> str: ...
        def _bind_state(self) -> None: ...
        def _is_real_status_reporter(self) -> bool: ...
        def _on_status_update(self, status: Any) -> None: ...
        def _check_cached_credentials(self, mode: str) -> bool: ...
        def _show_validation_screen(self, mode: str, callback: Any) -> None: ...
        def _continue_saved_mode_flow(self, mode: str, ok: bool) -> None: ...
        def _finish_saved_mode_setup(self, mode: str) -> None: ...
        def _handle_mode_selection(self, mode: str | None) -> None: ...
        def _start_readonly_data_feed(self) -> None: ...
        def push_screen(self, screen: Any, callback: Any = None) -> None: ...
        async def push_screen_wait(self, screen: Any) -> Any: ...
        def call_later(self, callback: Any) -> None: ...
        def log(self, message: str) -> None: ...
        def exit(self) -> None: ...

    async def on_mount(self: TraderApp) -> None:
        """Called when app starts."""
        logger.debug("TraderApp.on_mount() called")
        try:
            logger.debug("TUI mounting, initializing components")
            logger.debug(
                "Initial mode: %s, Bot available: %s",
                self._initial_mode or "mode selection",
                self.bot is not None,
            )

            # If bot provided directly (e.g., from CLI with --mode), use it
            if self.bot is not None:
                await self._initialize_with_bot()
                return

            # Check for saved mode preference (remember last mode)
            saved_mode = self.mode_service.load_mode_preference()
            if saved_mode:
                logger.debug("Found saved mode preference: %s", saved_mode)

                # For non-demo modes, check credential cache before full validation
                if saved_mode != "demo":
                    # Try cached validation first for quick resume
                    cache_valid = await self._check_cached_credentials(saved_mode)
                    if cache_valid:
                        # Cache hit - skip validation screen, show toast and proceed
                        logger.debug("Credential cache valid for %s, quick resume", saved_mode)
                        from gpt_trader.tui.notification_helpers import notify_success

                        notify_success(self, "Credentials verified ✓")
                        await self._finish_saved_mode_setup(saved_mode)
                        return
                    else:
                        # Cache miss - show full validation screen
                        self._show_validation_screen(
                            saved_mode,
                            lambda ok: self._continue_saved_mode_flow(saved_mode, ok),
                        )
                        return

                # Demo mode - proceed directly
                await self._finish_saved_mode_setup(saved_mode)
                return

            # No saved mode - show mode selection screen (first launch)
            from gpt_trader.tui.screens.mode_selection import ModeSelectionScreen

            logger.debug("No saved mode, showing mode selection screen")
            self.push_screen(ModeSelectionScreen(), callback=self._handle_mode_selection)

        except Exception as e:
            logger.critical(f"Failed to mount TUI: {e}", exc_info=True)
            notify_error(self, f"TUI initialization failed: {e}", timeout=30)
            raise

    async def _initialize_with_bot(self: TraderApp) -> None:
        """Initialize the TUI with a bot instance."""
        # Create WorkerService for off-UI-loop bot execution
        self.worker_service = WorkerService(self)

        # Create managers now that bot is available
        # Pass worker_service to ensure bot runs off the UI event loop
        self.lifecycle_manager = BotLifecycleManager(self, worker_service=self.worker_service)

        # Enable update throttling to batch high-frequency market data updates (100ms window)
        throttler = UpdateThrottler(min_interval=0.1)
        self.ui_coordinator = UICoordinator(self, throttler=throttler)
        logger.debug(
            "Created WorkerService, BotLifecycleManager and UICoordinator (throttling enabled)"
        )

        # Detect and set bot mode
        self.data_source_mode = self._detect_bot_mode()
        self.tui_state.data_source_mode = self.data_source_mode
        logger.debug("Bot mode detected: %s", self.data_source_mode)
        logger.debug(
            "System ready: StatusReporter available=%s",
            hasattr(self.bot.engine, "status_reporter") if self.bot else False,
        )

        # Show live mode warning if bot was provided directly and is live mode
        # (mode selection flow already showed warning before creating bot)
        if self.data_source_mode == "live" and self._initial_mode not in ("live", None):
            result = await self.push_screen_wait(LiveWarningModal())
            if not result:
                logger.info("User declined to continue in live mode")
                self.exit()
                return

        self.push_screen(MainScreen())
        self.log("TUI Started")

        # Handle missing StatusReporter with graceful degradation
        if (
            not hasattr(self.bot.engine, "status_reporter")
            or self.bot.engine.status_reporter is None
        ):
            logger.warning("StatusReporter not available - TUI will operate in degraded mode")
            from gpt_trader.tui.adapters.null_status_reporter import NullStatusReporter

            self.bot.engine.status_reporter = NullStatusReporter()
            self.tui_state.degraded_mode = True
            self.tui_state.degraded_reason = "StatusReporter not available"
            self.tui_state.connection_healthy = False
            notify_warning(
                self, "Limited functionality: Data source unavailable", title="Degraded Mode"
            )
        else:
            self.tui_state.degraded_mode = False
            self.tui_state.degraded_reason = ""

        # Note: Observer is connected in MainScreen.on_mount() to avoid race condition
        # where updates arrive before widgets are mounted. See connect_status_observer().

        # In read_only mode, auto-start data feed so users see market data immediately.
        # Trading controls (S key) remain for starting the actual trading bot.
        if self.data_source_mode == "read_only":
            notify_action(self, "Read-only mode - starting data feed...")
            self.tui_state.data_fetching = True
            self.call_later(self._start_readonly_data_feed)
            logger.debug("Auto-starting data feed for read_only mode")
        else:
            logger.debug("Bot initialized in STOPPED state. Press 's' to start.")

        # Start UI update loop (managed by UICoordinator)
        await self.ui_coordinator.start_update_loop()
        logger.debug("UI update loop started")

        # Bind state to widgets
        self._bind_state()

        # Initial UI sync will happen in MainScreen.on_mount() after widgets are ready
        logger.debug("TUI mounted successfully")

        # Initialize responsive state using ResponsiveManager
        self.responsive_state = self.responsive_manager.initialize(self.size.width)
        self.terminal_width = self.size.width

    async def on_unmount(self: TraderApp) -> None:
        """Called when app stops - ensure all cleanup happens."""
        try:
            logger.debug("TUI unmounting, cleaning up observers and tasks")

            # Detach log handler
            from gpt_trader.tui.log_manager import detach_tui_log_handler

            detach_tui_log_handler()
            logger.debug("TUI log handler detached")

            # Cleanup responsive manager
            self.responsive_manager.cleanup()

            # Cleanup managers (they handle their own task cancellation)
            if self.ui_coordinator:
                await self.ui_coordinator.stop_update_loop()
                logger.debug("UICoordinator stopped")

            if self.lifecycle_manager:
                self.lifecycle_manager.cleanup()
                logger.debug("BotLifecycleManager cleaned up")

            if self.worker_service:
                self.worker_service.cleanup()
                logger.debug("WorkerService cleaned up")

            # Stop bot if running
            if self.bot and self.bot.running:
                logger.info("Stopping bot")
                await self.bot.stop()

            # Remove observer (only for real StatusReporter, not NullStatusReporter)
            if self.bot and hasattr(self.bot.engine, "status_reporter"):
                if self._is_real_status_reporter():
                    self.bot.engine.status_reporter.remove_observer(self._on_status_update)
                    logger.debug("Removed StatusReporter observer")
                else:
                    logger.debug("Skipping observer removal for NullStatusReporter")

            # Cleanup bot resources (HTTP sessions, WebSocket connections)
            if self.bot:
                self._cleanup_bot_resources()

            # Clear performance service singleton
            from gpt_trader.tui.services import clear_tui_performance_service

            clear_tui_performance_service()
            logger.debug("Performance service cleared")

            # Clear container registry to prevent leaks between TUI sessions
            from gpt_trader.app.container import clear_application_container

            clear_application_container()
            logger.debug("Application container cleared")

            logger.debug("TUI unmounted successfully - all cleanup complete")
        except Exception as e:
            logger.error(f"Error during TUI unmount: {e}", exc_info=True)

    def _cleanup_bot_resources(self: TraderApp) -> None:
        """Clean up bot-owned resources like HTTP sessions and WebSocket connections.

        Called during unmount to prevent resource accumulation across TUI restarts.
        """
        if not self.bot:
            return

        try:
            # Close Coinbase client session if present
            client = getattr(self.bot, "client", None)
            if client is not None and hasattr(client, "close"):
                client.close()
                logger.debug("Coinbase client session closed")

            # Close WebSocket connection if present
            websocket = getattr(self.bot, "websocket", None)
            if websocket is not None and hasattr(websocket, "close"):
                websocket.close()
                logger.debug("WebSocket connection closed")

            # Also check engine for client/websocket
            engine = getattr(self.bot, "engine", None)
            if engine:
                engine_client = getattr(engine, "client", None)
                if engine_client is not None and hasattr(engine_client, "close"):
                    engine_client.close()
                    logger.debug("Engine client session closed")

                engine_ws = getattr(engine, "websocket", None)
                if engine_ws is not None and hasattr(engine_ws, "close"):
                    engine_ws.close()
                    logger.debug("Engine WebSocket closed")

        except Exception as e:
            logger.warning(f"Error during bot resource cleanup: {e}")

    def on_resize(self: TraderApp, event: events.Resize) -> None:
        """Handle terminal resize events with throttling.

        Delegates to ResponsiveManager for debounced handling.
        """
        self.responsive_manager.handle_resize(event.size.width)

    def watch_responsive_state(self: TraderApp, state: ResponsiveState) -> None:
        """Propagate responsive state changes to child widgets.

        Args:
            state: ResponsiveState enum value
        """
        # Update ResponsiveManager's tracked state
        self.responsive_manager.current_state = state
        # Propagate to screen
        self.responsive_manager.propagate_to_screen()

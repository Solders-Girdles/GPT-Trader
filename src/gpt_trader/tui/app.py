"""
Main TUI Application for GPT-Trader.

This is the primary entry point for the terminal user interface.
The app coordinates between various services and managers to provide
a complete trading experience.
"""

from __future__ import annotations

import asyncio
import json
import signal
import threading
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from textual import events
from textual.app import App
from textual.binding import Binding
from textual.command import Provider
from textual.reactive import reactive

from gpt_trader.monitoring.status_reporter import BotStatus
from gpt_trader.tui.commands import TraderCommands
from gpt_trader.tui.managers import BotLifecycleManager, UICoordinator
from gpt_trader.tui.notification_helpers import (
    notify_action,
    notify_error,
    notify_success,
    notify_warning,
)
from gpt_trader.tui.preferences_paths import resolve_preferences_paths
from gpt_trader.tui.responsive_state import ResponsiveState
from gpt_trader.tui.screens import DetailsScreen, MainScreen, MarketScreen
from gpt_trader.tui.services import (
    ActionDispatcher,
    AlertManager,
    ConfigService,
    ModeService,
    ResponsiveManager,
    StateRegistry,
    ThemeService,
    TuiPerformanceService,
    clear_tui_performance_service,
    set_tui_performance_service,
)
from gpt_trader.tui.services.mode_service import create_bot_for_mode
from gpt_trader.tui.services.worker_service import WorkerService
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets import LiveWarningModal, SlimStatusWidget
from gpt_trader.tui.widgets.error_indicator import ErrorIndicatorWidget
from gpt_trader.tui.widgets.status import BotStatusWidget
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.orchestration.trading_bot.bot import TradingBot

logger = get_logger(__name__, component="tui")


class TraderApp(App):
    """GPT-Trader Terminal User Interface.

    The main application class that coordinates all TUI components.
    Uses a service-oriented architecture to separate concerns:

    - ThemeService: Theme management and persistence
    - ConfigService: Configuration display
    - ResponsiveManager: Terminal resize handling
    - ModeService: Bot mode management
    - BotLifecycleManager: Bot start/stop operations
    - UICoordinator: UI update loop and state synchronization
    """

    # Use absolute path resolving relative to this file
    from pathlib import Path

    CSS_PATH = Path(__file__).parent / "styles" / "main.tcss"

    # Enable command palette with custom commands
    COMMANDS: set[type[Provider]] = {TraderCommands}

    # Responsive design properties
    terminal_width = reactive(120)
    responsive_state = reactive(ResponsiveState.STANDARD)

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "toggle_bot", "Start/Stop Bot"),
        ("c", "show_config", "Config"),
        ("l", "focus_logs", "Focus Logs"),
        ("m", "show_market", "Market"),  # Changed from mode_info
        ("d", "show_details", "Details"),  # New: Details overlay
        ("i", "show_mode_info", "Mode Info"),  # Moved from 'm' to 'i'
        ("r", "reconnect_data", "Reconnect"),
        ("p", "panic", "PANIC"),
        ("t", "toggle_theme", "Toggle Theme"),
        ("a", "show_alerts", "Alerts"),
        ("f", "force_refresh", "Refresh"),
        ("1", "show_full_logs", "Full Logs"),
        ("2", "show_system_details", "System"),
        ("?", "show_help", "Help"),
        # Command palette (Ctrl+K is more intuitive than default Ctrl+\)
        Binding("ctrl+k", "command_palette", "Commands", show=True),
        # Performance monitoring (Ctrl+P)
        Binding("ctrl+p", "toggle_performance", "Performance", show=False),
        # Log level shortcuts (hidden from footer, shown in help)
        Binding("ctrl+1", "set_log_level('DEBUG')", "Log: DEBUG", show=False),
        Binding("ctrl+2", "set_log_level('INFO')", "Log: INFO", show=False),
        Binding("ctrl+3", "set_log_level('WARNING')", "Log: WARN", show=False),
        Binding("ctrl+4", "set_log_level('ERROR')", "Log: ERROR", show=False),
    ]

    def __init__(
        self,
        bot: TradingBot | Any | None = None,
        initial_mode: str | None = None,
        demo_scenario: str = "mixed",
    ) -> None:
        """Initialize the TUI application.

        Args:
            bot: Optional bot instance (for backward compatibility with run command)
            initial_mode: Optional mode to start in. If None, shows mode selection screen
            demo_scenario: Scenario to use for demo mode
        """
        # Select CSS based on saved theme preference before App init.
        from gpt_trader.tui.theme import ThemeMode

        theme_mode = ThemeMode.DARK
        preferences_path, fallback_path = resolve_preferences_paths()
        for path in filter(None, (preferences_path, fallback_path)):
            try:
                if path.exists():
                    prefs = json.loads(path.read_text())
                    theme_mode = ThemeMode(prefs.get("theme", "dark"))
                    break
            except Exception:
                theme_mode = ThemeMode.DARK

        styles_dir = self.Path(__file__).parent / "styles"
        css_file = styles_dir / (
            "main_light.tcss" if theme_mode == ThemeMode.LIGHT else "main.tcss"
        )
        if not css_file.exists():
            # Prefer a functional UI over a hard crash if the light CSS hasn't
            # been generated yet.
            css_file = styles_dir / "main.tcss"

        super().__init__(css_path=css_file)
        self.bot = bot  # May be None if using mode selection flow
        self.data_source_mode: str = "demo"  # Will be set during on_mount
        self._initial_mode = initial_mode  # For mode selection flow
        self._demo_scenario = demo_scenario  # For demo mode

        # Initialize State
        self.tui_state: TuiState = TuiState()

        # Initialize error tracker widget (singleton for whole app)
        self.error_tracker: ErrorIndicatorWidget = ErrorIndicatorWidget(max_errors=10)

        # Startup bootstrap: fetch balances/market snapshot even while STOPPED.
        self._bootstrap_snapshot_requested: bool = False
        self._bootstrap_snapshot_inflight: bool = False

        # Initialize services
        self.theme_service = ThemeService(self)
        self.config_service = ConfigService(self)
        self.responsive_manager = ResponsiveManager(self)
        self.mode_service = ModeService(self, demo_scenario=demo_scenario)
        self.action_dispatcher = ActionDispatcher(self)
        self.alert_manager = AlertManager(self)
        self.state_registry = StateRegistry()

        # Initialize performance monitoring service
        self.performance_service = TuiPerformanceService(app=self, enabled=True)
        set_tui_performance_service(self.performance_service)

        # Load saved theme preference
        self.theme_service.load_preference()

        # Create managers (only after bot is set)
        self.lifecycle_manager: BotLifecycleManager | None = None
        self.ui_coordinator: UICoordinator | None = None
        self.worker_service: WorkerService | None = None

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def apply_theme_css(self, theme_mode: Any) -> bool:
        """Hot-swap the active CSS file for the requested theme mode."""
        from gpt_trader.tui.theme import ThemeMode

        mode = theme_mode if isinstance(theme_mode, ThemeMode) else ThemeMode(str(theme_mode))
        styles_dir = self.Path(__file__).parent / "styles"

        # Map theme modes to CSS files
        css_file_map = {
            ThemeMode.DARK: "main.tcss",
            ThemeMode.LIGHT: "main_light.tcss",
            ThemeMode.HIGH_CONTRAST: "main_high_contrast.tcss",
        }
        css_file = styles_dir / css_file_map.get(mode, "main.tcss")
        if not css_file.exists():
            return False

        theme_css_paths = {
            str((styles_dir / "main.tcss").absolute()),
            str((styles_dir / "main_light.tcss").absolute()),
            str((styles_dir / "main_high_contrast.tcss").absolute()),
        }

        # Remove any previously-loaded theme CSS sources so switching doesn't
        # accumulate duplicate rules over time.
        for key in list(self.stylesheet.source.keys()):
            path, _scope = key
            if path in theme_css_paths:
                self.stylesheet.source.pop(key, None)

        self.stylesheet.read(css_file)
        self.refresh_css(animate=False)
        return True

    def _detect_bot_mode(self) -> str:
        """Detect current bot operating mode.

        Delegates to ModeService or BotLifecycleManager.
        """
        if self.bot:
            return self.mode_service.detect_bot_mode(self.bot)
        if self.lifecycle_manager:
            return self.lifecycle_manager.detect_bot_mode()
        return "demo"  # Default fallback

    async def _switch_to_mode(self, target_mode: str) -> bool:
        """Switch to a new bot mode safely.

        Delegates to BotLifecycleManager.
        """
        if self.lifecycle_manager:
            return await self.lifecycle_manager.switch_mode(target_mode)
        return False

    async def _start_readonly_data_feed(self) -> None:
        """Auto-start data feed for read-only mode.

        Called via call_later() after initialization to fetch initial data
        without requiring the user to press 'S'.
        """
        try:
            logger.debug("Starting read-only data feed")

            # Request bootstrap snapshot for initial data
            request_bootstrap = getattr(self, "request_bootstrap_snapshot", None)
            if callable(request_bootstrap):
                request_bootstrap(force=True)

            # Sync state from StatusReporter
            if self.ui_coordinator:
                self.ui_coordinator.sync_state_from_bot()

            self.tui_state.data_fetching = False
            logger.info("Read-only data feed started successfully")
        except Exception as e:
            self.tui_state.data_fetching = False
            logger.error(f"Failed to start read-only data feed: {e}", exc_info=True)
            notify_error(self, f"Failed to start data feed: {e}", title="Error")

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle termination signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        # Schedule the exit on the event loop
        if hasattr(self, "exit"):
            self.exit()

    async def on_mount(self) -> None:
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

    async def _handle_mode_selection(self, selected_mode: str | None) -> None:
        """Handle mode selection from the mode selection screen."""
        if selected_mode is None:
            logger.info("User cancelled mode selection")
            self.exit()
            return

        # Handle setup wizard request from mode selection
        if selected_mode == "setup":
            from gpt_trader.tui.screens.api_setup_wizard import APISetupWizardScreen
            from gpt_trader.tui.screens.mode_selection import ModeSelectionScreen

            logger.info("User requested API setup wizard from mode selection")

            def handle_wizard_complete(result: str | None) -> None:
                """Return to mode selection after wizard completes."""
                if result == "verify":
                    self.notify("Credentials saved! Select a mode to continue.", timeout=5)
                self.push_screen(ModeSelectionScreen(), callback=self._handle_mode_selection)

            self.push_screen(APISetupWizardScreen(), callback=handle_wizard_complete)
            return

        # Validate credentials for non-demo modes
        if selected_mode != "demo":
            # Use callback-based validation flow (can't use await with push_screen_wait here)
            self._show_validation_screen(
                selected_mode,
                lambda ok: self._continue_mode_selection_flow(selected_mode, ok),
            )
            return

        # Demo mode - proceed directly without validation
        await self._finish_mode_selection_setup(selected_mode)

    async def _initialize_with_bot(self) -> None:
        """Initialize the TUI with a bot instance."""
        # Create WorkerService for off-UI-loop bot execution
        self.worker_service = WorkerService(self)

        # Create managers now that bot is available
        # Pass worker_service to ensure bot runs off the UI event loop
        self.lifecycle_manager = BotLifecycleManager(self, worker_service=self.worker_service)
        self.ui_coordinator = UICoordinator(self)
        logger.debug("Created WorkerService, BotLifecycleManager and UICoordinator")

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

    def _show_validation_screen(
        self,
        mode: str,
        on_complete: callable,
    ) -> None:
        """Show credential validation screen with callback.

        This method runs validation in a worker and shows results in a modal.

        Args:
            mode: The trading mode to validate for.
            on_complete: Callback with signature (should_proceed: bool) -> None
        """
        from gpt_trader.tui.screens.api_setup_wizard import APISetupWizardScreen
        from gpt_trader.tui.screens.credential_validation_screen import (
            CredentialValidationScreen,
        )
        from gpt_trader.tui.services.credential_validator import CredentialValidator

        async def do_validation() -> None:
            logger.debug("Validating credentials for mode: %s", mode)
            validator = CredentialValidator(self)
            result = await validator.validate_for_mode(mode)

            # Show validation screen and get result via callback
            def handle_validation_result(should_proceed: bool | str | None) -> None:
                if should_proceed == "setup":
                    # User wants to launch the API key setup wizard
                    logger.info("User requested API key setup wizard")
                    self.push_screen(
                        APISetupWizardScreen(),
                        callback=lambda wizard_result: self._handle_wizard_result(
                            mode, wizard_result, on_complete
                        ),
                    )
                elif should_proceed == "retry":
                    # User wants to retry validation (e.g., after fixing env vars)
                    logger.info(f"User requested retry validation for {mode} mode")
                    notify_action(self, "Retrying validation...")
                    self._show_validation_screen(mode, on_complete)
                elif should_proceed:
                    logger.info(f"Credential validation passed for {mode} mode")
                    # Cache successful validation for quick resume on next launch
                    self._cache_credential_validation(mode)
                    on_complete(True)
                else:
                    logger.info(f"User cancelled credential validation for {mode} mode")
                    on_complete(False)

            self.push_screen(
                CredentialValidationScreen(result),
                callback=handle_validation_result,
            )

        self.run_worker(do_validation(), exclusive=True)

    async def _check_cached_credentials(self, mode: str) -> bool:
        """Check if we have valid cached credentials for the mode.

        Returns True if cache is valid and we can skip validation screen.

        Args:
            mode: Trading mode to check cache for.

        Returns:
            True if cached credentials are valid for this mode.
        """
        from gpt_trader.tui.services.credential_validator import CredentialValidator
        from gpt_trader.tui.services.preferences_service import get_preferences_service

        validator = CredentialValidator(self)
        prefs = get_preferences_service()

        # Compute current fingerprint
        current_fp = validator.compute_credential_fingerprint()
        if not current_fp:
            logger.debug("No credential fingerprint available")
            return False

        # Check cache validity
        is_valid = prefs.is_credential_cache_valid(current_fp, mode)
        return is_valid

    def _cache_credential_validation(self, mode: str) -> None:
        """Cache successful credential validation for quick resume.

        Args:
            mode: The mode that was successfully validated.
        """
        from gpt_trader.tui.services.credential_validator import CredentialValidator
        from gpt_trader.tui.services.preferences_service import get_preferences_service

        validator = CredentialValidator(self)
        prefs = get_preferences_service()

        fingerprint = validator.compute_credential_fingerprint()
        if fingerprint:
            # Get existing validation modes and add this one
            cache = prefs.get_credential_cache()
            validation_modes = cache.get("validation_modes", {})
            validation_modes[mode] = True
            prefs.set_credential_cache(fingerprint, validation_modes)
            logger.debug("Cached credential validation for '%s' mode", mode)

    def _handle_wizard_result(
        self,
        mode: str,
        wizard_result: str | None,
        on_complete: callable,
    ) -> None:
        """Handle result from the API setup wizard.

        Args:
            mode: The trading mode being validated.
            wizard_result: Result from wizard ("verify" or None if cancelled).
            on_complete: Original callback to invoke after validation.
        """
        if wizard_result == "verify":
            # User completed wizard - re-run validation
            logger.debug("Re-validating credentials for %s after wizard completion", mode)
            self._show_validation_screen(mode, on_complete)
        else:
            # User cancelled wizard - return to mode selection
            logger.info("User cancelled setup wizard")
            on_complete(False)

    def _continue_saved_mode_flow(self, mode: str, validation_ok: bool) -> None:
        """Continue saved mode flow after credential validation.

        Args:
            mode: The saved mode being restored.
            validation_ok: Whether validation passed and user wants to proceed.
        """
        if validation_ok:
            # Validation passed - proceed with saved mode setup
            asyncio.create_task(self._finish_saved_mode_setup(mode))
        else:
            # Validation failed or user cancelled - show mode selection
            from gpt_trader.tui.screens.mode_selection import ModeSelectionScreen

            logger.warning("Saved mode validation failed, showing mode selection")
            self.push_screen(ModeSelectionScreen(), callback=self._handle_mode_selection)

    async def _finish_saved_mode_setup(self, mode: str) -> None:
        """Finish setting up a saved mode after validation.

        Args:
            mode: The validated mode to initialize.
        """
        # Show live warning if needed
        if mode == "live":
            should_continue = await self.mode_service.show_live_warning()
            if not should_continue:
                from gpt_trader.tui.screens.mode_selection import ModeSelectionScreen

                logger.info("User declined live mode, showing mode selection")
                self.push_screen(ModeSelectionScreen(), callback=self._handle_mode_selection)
                return

        # Create bot for saved mode
        logger.debug("Creating bot for saved mode: %s", mode)
        self.bot = create_bot_for_mode(mode, self._demo_scenario)

        # Initialize with the bot
        await self._initialize_with_bot()

    def _continue_mode_selection_flow(self, selected_mode: str, validation_ok: bool) -> None:
        """Continue mode selection flow after credential validation.

        Args:
            selected_mode: The mode selected by user.
            validation_ok: Whether validation passed and user wants to proceed.
        """
        if validation_ok:
            # Validation passed - proceed with mode setup
            asyncio.create_task(self._finish_mode_selection_setup(selected_mode))
        else:
            # Validation failed or user cancelled - return to mode selection
            from gpt_trader.tui.screens.mode_selection import ModeSelectionScreen

            logger.warning("Validation failed, returning to mode selection")
            self.push_screen(ModeSelectionScreen(), callback=self._handle_mode_selection)

    async def _finish_mode_selection_setup(self, selected_mode: str) -> None:
        """Finish mode selection setup after validation.

        Args:
            selected_mode: The validated mode to initialize.
        """
        # Show live warning before creating bot
        if selected_mode == "live":
            should_continue = await self.mode_service.show_live_warning()
            if not should_continue:
                logger.info("User declined to continue in live mode")
                self.exit()
                return

        # Save mode preference for future launches
        self.mode_service.save_mode_preference(selected_mode)

        # Create bot for selected mode using ModeService
        logger.debug("Creating bot for selected mode: %s", selected_mode)
        self.bot = create_bot_for_mode(selected_mode, self._demo_scenario)

        # Initialize with the newly created bot
        await self._initialize_with_bot()

    def on_resize(self, event: events.Resize) -> None:
        """Handle terminal resize events with throttling.

        Delegates to ResponsiveManager for debounced handling.
        """
        self.responsive_manager.handle_resize(event.size.width)

    def watch_responsive_state(self, state: ResponsiveState) -> None:
        """Propagate responsive state changes to child widgets.

        Args:
            state: ResponsiveState enum value
        """
        # Update ResponsiveManager's tracked state
        self.responsive_manager.current_state = state
        # Propagate to screen
        self.responsive_manager.propagate_to_screen()

    async def on_unmount(self) -> None:
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
            clear_tui_performance_service()
            logger.debug("Performance service cleared")

            logger.debug("TUI unmounted successfully - all cleanup complete")
        except Exception as e:
            logger.error(f"Error during TUI unmount: {e}", exc_info=True)

    def _cleanup_bot_resources(self) -> None:
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

    def _on_status_update(self, status: BotStatus) -> None:
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

    def _apply_status_update(self, status: BotStatus) -> None:
        """Backward-compatible alias for tests/legacy callers."""
        self._on_status_update(status)

    def _bind_state(self) -> None:
        """Bind reactive state to widgets."""
        # This is where we could set up direct bindings if widgets supported it
        # For now, we'll just rely on the update loop pushing data to state,
        # and then we can push state to widgets or have widgets watch state.
        pass

    def _is_real_status_reporter(self) -> bool:
        """Check if the current status_reporter is a real reporter (not NullStatusReporter).

        Returns:
            True if it's a real reporter that provides data, False if it's a null adapter.
        """
        if not self.bot or not hasattr(self.bot.engine, "status_reporter"):
            return False
        reporter = self.bot.engine.status_reporter
        return not getattr(reporter, "is_null_reporter", False)

    def connect_status_observer(self) -> None:
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

        self.bot.engine.status_reporter.add_observer(self._on_status_update)
        logger.debug("Connected to StatusReporter observer (from MainScreen.on_mount)")

    def _sync_state_from_bot(self) -> None:
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

    def request_bootstrap_snapshot(self, *, force: bool = False) -> None:
        """Request a one-time bootstrap snapshot in the background.

        This is used on startup to populate account balances and a basic market
        snapshot even when the bot is STOPPED (manual-start policy).
        """
        if self.data_source_mode == "demo":
            return

        # Avoid scheduling bootstrap work when there is no real broker attached.
        # This also prevents MagicMock auto-attribute creation in unit tests.
        try:
            if self.bot is None:
                return
            bot_dict = getattr(self.bot, "__dict__", None)
            if isinstance(bot_dict, dict) and bot_dict.get("broker") is None:
                return
        except Exception:
            pass

        if self._bootstrap_snapshot_inflight:
            return

        if self._bootstrap_snapshot_requested and not force:
            return

        self._bootstrap_snapshot_requested = True

        if not self.worker_service:
            return

        async def fetch() -> None:
            await self.bootstrap_snapshot()

        self._bootstrap_snapshot_inflight = True
        self.worker_service.run_data_fetch(fetch, name="bootstrap_snapshot")

    async def bootstrap_snapshot(self) -> bool:
        """Fetch balances + a minimal market snapshot and push into the UI.

        Returns:
            True if any data was fetched and applied.
        """
        # Avoid doing work in demo mode or while bot is running.
        if self.data_source_mode == "demo":
            self._bootstrap_snapshot_inflight = False
            return False

        if self.bot is None:
            self._bootstrap_snapshot_inflight = False
            return False

        if bool(getattr(self.bot, "running", False)):
            self._bootstrap_snapshot_inflight = False
            return False

        # Avoid MagicMock attr auto-creation in tests by preferring __dict__.
        broker = None
        try:
            bot_dict = getattr(self.bot, "__dict__", {})
            broker = bot_dict.get("broker") if isinstance(bot_dict, dict) else None
        except Exception:
            broker = None

        # Fallback only if the bot doesn't expose __dict__ (avoid MagicMock attr creation).
        if broker is None and not isinstance(getattr(self.bot, "__dict__", None), dict):
            try:
                broker = getattr(self.bot, "broker", None)
            except Exception:
                broker = None

        if broker is None:
            self._bootstrap_snapshot_inflight = False
            return False

        # StatusReporter is the source of truth for sync_state_from_bot().
        reporter = None
        try:
            reporter = getattr(getattr(self.bot, "engine", None), "status_reporter", None)
        except Exception:
            reporter = None

        if reporter is None or getattr(reporter, "is_null_reporter", False):
            self._bootstrap_snapshot_inflight = False
            return False

        from gpt_trader.core import Balance as CoreBalance
        from gpt_trader.tui.formatting import safe_decimal

        stable_assets = {"USD", "USDC", "USDT", "DAI"}

        def fetch_sync() -> tuple[list[CoreBalance], dict[str, Decimal], Decimal]:
            """Run blocking broker calls in a single thread for session safety."""
            balances: list[CoreBalance] = []
            prices: dict[str, Decimal] = {}

            # 1) Balances
            try:
                balances = list(broker.list_balances() or [])
            except Exception as exc:
                raise RuntimeError(f"Failed to fetch balances: {exc}") from exc

            non_zero = [b for b in balances if getattr(b, "total", Decimal("0")) > 0]

            # 2) Market prices: bot symbols first, then holdings (cap for sanity)
            symbols: list[str] = []
            seen: set[str] = set()

            try:
                cfg_symbols = list(getattr(getattr(self.bot, "config", None), "symbols", []) or [])
            except Exception:
                cfg_symbols = []

            for sym in cfg_symbols:
                sym_str = str(sym)
                if sym_str and sym_str not in seen:
                    seen.add(sym_str)
                    symbols.append(sym_str)

            for bal in non_zero:
                asset = str(getattr(bal, "asset", "") or "").upper()
                if not asset or asset in stable_assets:
                    continue
                for quote in ("USD", "USDC"):
                    product_id = f"{asset}-{quote}"
                    if product_id not in seen:
                        seen.add(product_id)
                        symbols.append(product_id)
                        break

            # Hard cap to avoid excessive HTTP calls on startup.
            symbols = symbols[:25]

            for product_id in symbols:
                try:
                    ticker = broker.get_ticker(product_id) or {}
                    price = safe_decimal(
                        ticker.get("price")
                        or ticker.get("last")
                        or ticker.get("trade_price")
                        or "0"
                    )
                    if price > 0:
                        prices[product_id] = price
                except Exception:
                    continue

            # 3) Estimate equity in USD using stable balances + known tickers.
            equity = Decimal("0")
            for bal in non_zero:
                asset = str(getattr(bal, "asset", "") or "").upper()
                total = safe_decimal(getattr(bal, "total", "0"))
                if total <= 0:
                    continue
                if asset in stable_assets:
                    equity += total
                    continue

                usd_price = prices.get(f"{asset}-USD")
                if usd_price is None:
                    usdc_price = prices.get(f"{asset}-USDC")
                    usd_price = usdc_price
                if usd_price is not None and usd_price > 0:
                    equity += total * usd_price

            return non_zero, prices, equity

        try:
            balances, prices, equity = await asyncio.to_thread(fetch_sync)
        except Exception as e:
            logger.warning("Bootstrap snapshot failed: %s", e)
            try:
                notify_warning(self, f"Failed to fetch account snapshot: {e}", title="Startup")
            except Exception:
                pass
            self._bootstrap_snapshot_inflight = False
            return False

        # Apply to StatusReporter (so subsequent syncs preserve it).
        try:
            reporter.update_account(balances, summary={})
            reporter.update_equity(equity)
            for symbol, price in prices.items():
                reporter.update_price(symbol, price)
        except Exception as e:
            logger.debug(
                "Failed applying bootstrap snapshot to StatusReporter: %s", e, exc_info=True
            )

        # Refresh TuiState/UI from StatusReporter.
        try:
            if self.ui_coordinator:
                self.ui_coordinator.sync_state_from_bot()
                self.ui_coordinator.update_main_screen()
            else:
                self._sync_state_from_bot()
        except Exception:
            pass

        self._bootstrap_snapshot_inflight = False
        return bool(balances or prices)

    def _pulse_heartbeat(self) -> None:
        """Smooth heartbeat pulse using sine wave."""
        import math
        import time

        try:
            status_widget = self.query_one(BotStatusWidget)

            # Calculate sine wave: 0.0 to 1.0
            t = time.time()
            pulse = (math.sin(t * 2) + 1) / 2  # Sine wave normalized to 0-1

            status_widget.heartbeat = pulse
        except Exception as e:
            logger.debug(f"Failed to pulse heartbeat: {e}")

    # =========================================================================
    # Action Handlers (delegated to ActionDispatcher)
    # =========================================================================

    async def action_toggle_bot(self) -> None:
        """Toggle bot running state."""
        await self.action_dispatcher.toggle_bot()

    async def action_show_config(self) -> None:
        """Show configuration modal."""
        await self.action_dispatcher.show_config()

    async def action_focus_logs(self) -> None:
        """Focus the log widget."""
        await self.action_dispatcher.focus_logs()

    def action_set_log_level(self, level: str) -> None:
        """Set log level via keyboard shortcut.

        Args:
            level: Log level name (DEBUG, INFO, WARNING, ERROR)
        """
        from gpt_trader.tui.widgets.logs import LogWidget

        level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        try:
            log_widget = self.query_one("#dash-logs", LogWidget)
            log_widget.set_level(level_map.get(level, 20))
            notify_action(self, f"Log level: {level}")
        except Exception:
            pass

    async def action_show_full_logs(self) -> None:
        """Show full logs screen."""
        await self.action_dispatcher.show_full_logs()

    async def action_show_system_details(self) -> None:
        """Show system details screen."""
        await self.action_dispatcher.show_system_details()

    async def action_show_mode_info(self) -> None:
        """Show mode information modal."""
        await self.action_dispatcher.show_mode_info()

    async def action_show_market(self) -> None:
        """Show market overlay screen."""
        self.push_screen(MarketScreen())

    async def action_show_details(self) -> None:
        """Show details overlay screen."""
        self.push_screen(DetailsScreen())

    async def action_show_help(self) -> None:
        """Show help screen."""
        await self.action_dispatcher.show_help()

    async def action_reconnect_data(self) -> None:
        """Reconnect data source."""
        await self.action_dispatcher.reconnect_data()

    async def action_panic(self) -> None:
        """Show panic confirmation modal."""
        await self.action_dispatcher.panic()

    async def action_toggle_theme(self) -> None:
        """Toggle theme."""
        await self.action_dispatcher.toggle_theme()

    async def action_show_alerts(self) -> None:
        """Show alert history screen."""
        await self.action_dispatcher.show_alert_history()

    async def action_force_refresh(self) -> None:
        """Force refresh bot state and UI."""
        await self.action_dispatcher.force_refresh()

    async def action_toggle_performance(self) -> None:
        """Toggle performance monitoring overlay.

        Shows real-time TUI performance metrics including FPS, latency,
        memory usage, and throttler efficiency.
        """
        from gpt_trader.tui.screens.performance_screen import PerformanceScreen

        try:
            # Check if already showing performance screen
            self.query_one(PerformanceScreen)
            self.pop_screen()
        except Exception:
            # Not showing - push performance screen
            self.push_screen(PerformanceScreen())

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_bot_status_widget_toggle_bot_pressed(
        self, message: BotStatusWidget.ToggleBotPressed
    ) -> None:
        """Handle start/stop button press from BotStatusWidget."""
        asyncio.create_task(self.action_toggle_bot())

    def on_slim_status_widget_toggle_bot_pressed(
        self, message: SlimStatusWidget.ToggleBotPressed
    ) -> None:
        """Handle start/stop button press from SlimStatusWidget."""
        asyncio.create_task(self.action_toggle_bot())

    def on_slim_status_widget_mode_changed(self, message: SlimStatusWidget.ModeChanged) -> None:
        """Handle mode change request from SlimStatusWidget."""
        asyncio.create_task(self._switch_to_mode(message.mode))

    def on_mode_selector_mode_changed(self, message: Any) -> None:
        """Handle mode change request from ModeSelector."""
        from gpt_trader.tui.widgets import ModeSelector

        if isinstance(message, ModeSelector.ModeChanged):
            asyncio.create_task(self._switch_to_mode(message.new_mode))

    async def action_quit(self) -> None:
        """Quit the application."""
        await self.action_dispatcher.quit_app()

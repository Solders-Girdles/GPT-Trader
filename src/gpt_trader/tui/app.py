"""
Main TUI Application for GPT-Trader.

This is the primary entry point for the terminal user interface.
The app coordinates between various services and managers to provide
a complete trading experience.
"""

from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING, Any

from textual import events
from textual.app import App
from textual.binding import Binding
from textual.reactive import reactive

from gpt_trader.monitoring.status_reporter import BotStatus
from gpt_trader.tui.managers import BotLifecycleManager, UICoordinator
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
        super().__init__()
        self.bot = bot  # May be None if using mode selection flow
        self.data_source_mode: str = "demo"  # Will be set during on_mount
        self._initial_mode = initial_mode  # For mode selection flow
        self._demo_scenario = demo_scenario  # For demo mode

        # Initialize State
        self.tui_state: TuiState = TuiState()

        # Initialize error tracker widget (singleton for whole app)
        self.error_tracker: ErrorIndicatorWidget = ErrorIndicatorWidget(max_errors=10)

        # Initialize services
        self.theme_service = ThemeService(self)
        self.config_service = ConfigService(self)
        self.responsive_manager = ResponsiveManager(self)
        self.mode_service = ModeService(self, demo_scenario=demo_scenario)
        self.action_dispatcher = ActionDispatcher(self)
        self.alert_manager = AlertManager(self)
        self.state_registry = StateRegistry()

        # Load saved theme preference
        self.theme_service.load_preference()

        # Create managers (only after bot is set)
        self.lifecycle_manager: BotLifecycleManager | None = None
        self.ui_coordinator: UICoordinator | None = None
        self.worker_service: WorkerService | None = None

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

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
            logger.info("TUI mounting, initializing components")

            # If bot provided directly (e.g., from CLI with --mode), use it
            if self.bot is not None:
                await self._initialize_with_bot()
                return

            # Check for saved mode preference (remember last mode)
            saved_mode = self.mode_service.load_mode_preference()
            if saved_mode:
                logger.info(f"Found saved mode preference: {saved_mode}")

                # Validate credentials for saved non-demo modes
                if saved_mode != "demo":
                    # Use callback-based validation flow
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

            logger.info("No saved mode, showing mode selection screen")
            self.push_screen(ModeSelectionScreen(), callback=self._handle_mode_selection)

        except Exception as e:
            logger.critical(f"Failed to mount TUI: {e}", exc_info=True)
            self.notify(f"TUI initialization failed: {e}", severity="error", timeout=30)
            raise

    async def _handle_mode_selection(self, selected_mode: str | None) -> None:
        """Handle mode selection from the mode selection screen."""
        if selected_mode is None:
            logger.info("User cancelled mode selection")
            self.exit()
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
        logger.info("Created WorkerService, BotLifecycleManager and UICoordinator")

        # Detect and set bot mode
        self.data_source_mode = self._detect_bot_mode()
        self.tui_state.data_source_mode = self.data_source_mode
        logger.info(f"Bot mode detected: {self.data_source_mode}")

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

        # Connect to StatusReporter observer (required for TUI)
        if not hasattr(self.bot.engine, "status_reporter"):
            logger.critical(
                "StatusReporter not available - TUI requires StatusReporter for operation"
            )
            raise RuntimeError(
                "TUI requires bot.engine.status_reporter for data updates. "
                "Ensure the bot engine has StatusReporter properly initialized."
            )

        self.bot.engine.status_reporter.add_observer(self._on_status_update)
        logger.info("Connected to StatusReporter observer")

        # DON'T auto-start the bot - let user press 's' to start when ready
        logger.info("Bot initialized in STOPPED state. Press 's' to start.")

        # Start UI update loop (managed by UICoordinator)
        await self.ui_coordinator.start_update_loop()
        logger.info("UI update loop started")

        # Bind state to widgets
        self._bind_state()

        # Initial UI sync will happen in MainScreen.on_mount() after widgets are ready
        logger.info("TUI mounted successfully")

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
            logger.info(f"Validating credentials for mode: {mode}")
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
                elif should_proceed:
                    logger.info(f"Credential validation passed for {mode} mode")
                    on_complete(True)
                else:
                    logger.info(f"User cancelled credential validation for {mode} mode")
                    on_complete(False)

            self.push_screen(
                CredentialValidationScreen(result),
                callback=handle_validation_result,
            )

        self.run_worker(do_validation(), exclusive=True)

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
            logger.info(f"Re-validating credentials for {mode} after wizard completion")
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

            logger.info("Saved mode validation failed, showing mode selection")
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
        logger.info(f"Creating bot for saved mode: {mode}")
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

            logger.info("Validation failed, returning to mode selection")
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
        logger.info(f"Creating bot for selected mode: {selected_mode}")
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
            logger.info("TUI unmounting, cleaning up observers and tasks")

            # Detach log handler
            from gpt_trader.tui.log_manager import detach_tui_log_handler

            detach_tui_log_handler()
            logger.info("TUI log handler detached")

            # Cleanup responsive manager
            self.responsive_manager.cleanup()

            # Cleanup managers (they handle their own task cancellation)
            if self.ui_coordinator:
                await self.ui_coordinator.stop_update_loop()
                logger.info("UICoordinator stopped")

            if self.lifecycle_manager:
                self.lifecycle_manager.cleanup()
                logger.info("BotLifecycleManager cleaned up")

            if self.worker_service:
                self.worker_service.cleanup()
                logger.info("WorkerService cleaned up")

            # Stop bot if running
            if self.bot and self.bot.running:
                logger.info("Stopping bot")
                await self.bot.stop()

            # Remove observer
            if self.bot and hasattr(self.bot.engine, "status_reporter"):
                self.bot.engine.status_reporter.remove_observer(self._on_status_update)
                logger.info("Removed StatusReporter observer")

            logger.info("TUI unmounted successfully - all cleanup complete")
        except Exception as e:
            logger.error(f"Error during TUI unmount: {e}", exc_info=True)

    def _on_status_update(self, status: BotStatus) -> None:
        """Callback for StatusReporter updates (receives typed BotStatus).

        Delegates to UICoordinator via call_from_thread for thread safety.
        """
        if self.ui_coordinator:
            # This might be called from a background thread or loop
            # Schedule the update on the main thread
            self.call_from_thread(self.ui_coordinator.apply_observer_update, status)

    def _bind_state(self) -> None:
        """Bind reactive state to widgets."""
        # This is where we could set up direct bindings if widgets supported it
        # For now, we'll just rely on the update loop pushing data to state,
        # and then we can push state to widgets or have widgets watch state.
        pass

    def _sync_state_from_bot(self) -> None:
        """Manually sync state from bot (delegates to UICoordinator when available).

        Called by BotLifecycleManager on bot start/stop/mode-switch.
        Falls back to direct state update when ui_coordinator is None (e.g., in tests).
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

            # Access StatusReporter for typed data
            if hasattr(self.bot, "engine") and hasattr(self.bot.engine, "status_reporter"):
                status = self.bot.engine.status_reporter.get_status()
                self.tui_state.update_from_bot_status(status, runtime_state)

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
            self.notify(f"Log level: {level}", timeout=2)
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

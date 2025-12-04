"""
Main TUI Application for GPT-Trader.
"""

from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING, Any

from textual import events
from textual.app import App
from textual.reactive import reactive

from gpt_trader.monitoring.status_reporter import BotStatus
from gpt_trader.tui.responsive import calculate_responsive_state
from gpt_trader.tui.screens import FullLogsScreen, MainScreen, SystemDetailsScreen
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets import ConfigModal, LiveWarningModal, ModeInfoModal
from gpt_trader.tui.widgets.status import BotStatusWidget
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.orchestration.trading_bot.bot import TradingBot

logger = get_logger(__name__, component="tui")


def _create_bot_for_mode(mode: str, demo_scenario: str = "mixed") -> Any:
    """
    Create a bot instance for the specified mode.

    Args:
        mode: One of "demo", "paper", "read_only", "live"
        demo_scenario: Scenario to use for demo mode

    Returns:
        Bot instance (DemoBot or TradingBot)
    """
    if mode == "demo":
        from gpt_trader.tui.demo.demo_bot import DemoBot
        from gpt_trader.tui.demo.scenarios import get_scenario

        scenario = get_scenario(demo_scenario)
        return DemoBot(data_generator=scenario)
    else:
        from gpt_trader.cli.services import instantiate_bot, load_config_from_yaml
        from gpt_trader.orchestration.configuration import BotConfig, Profile

        if mode == "paper":
            try:
                config = load_config_from_yaml("config/profiles/paper.yaml")
            except Exception:
                config = BotConfig.from_profile(profile=Profile.DEMO, mock_broker=False)
        elif mode == "read_only":
            try:
                config = load_config_from_yaml("config/profiles/observe.yaml")
                config.read_only = True
            except Exception:
                config = BotConfig.from_profile(profile=Profile.DEMO, mock_broker=False)
                config.read_only = True
        elif mode == "live":
            try:
                config = load_config_from_yaml("config/profiles/prod.yaml")
            except Exception:
                config = BotConfig.from_profile(profile=Profile.PROD, mock_broker=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return instantiate_bot(config)


class TraderApp(App):
    """GPT-Trader Terminal User Interface."""

    CSS_PATH = "styles/main.tcss"

    # Responsive design properties
    terminal_width = reactive(120)
    responsive_state = reactive("standard")

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "toggle_bot", "Start/Stop Bot"),
        ("c", "show_config", "Config"),
        ("l", "focus_logs", "Focus Logs"),
        ("m", "show_mode_info", "Mode Info"),
        ("r", "reconnect_data", "Reconnect"),
        ("p", "panic", "PANIC"),
        ("1", "show_full_logs", "Full Logs"),
        ("2", "show_system_details", "System"),
    ]

    def __init__(
        self,
        bot: TradingBot | Any | None = None,
        initial_mode: str | None = None,
        demo_scenario: str = "mixed",
    ) -> None:
        """
        Initialize the TUI application.

        Args:
            bot: Optional bot instance (for backward compatibility with run command)
            initial_mode: Optional mode to start in. If None, shows mode selection screen
            demo_scenario: Scenario to use for demo mode
        """
        super().__init__()
        self.bot = bot  # May be None if using mode selection flow
        self._bot_task: asyncio.Task | None = None  # Track bot task for proper cancellation
        self._update_task: asyncio.Task | None = None
        self.data_source_mode: str = "demo"  # Will be set during on_mount
        self._initial_mode = initial_mode  # For mode selection flow
        self._demo_scenario = demo_scenario  # For demo mode

        # Initialize State
        self.tui_state: TuiState = TuiState()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _detect_bot_mode(self) -> str:
        """
        Detect current bot operating mode.

        Returns:
            "demo" - Mock data simulation
            "paper" - Real data with simulated execution
            "read_only" - Real data observation, no execution
            "live" - Real trading with real execution
        """
        # Check bot type first
        if type(self.bot).__name__ == "DemoBot":
            return "demo"

        # Check for read-only mode in config
        if hasattr(self.bot, "config") and getattr(self.bot.config, "read_only", False):
            return "read_only"

        # Check broker type
        if hasattr(self.bot, "engine") and hasattr(self.bot.engine, "broker"):
            broker_type = type(self.bot.engine.broker).__name__
            if broker_type == "HybridPaperBroker":
                return "paper"

        # Default to live if we have a real bot but don't match other modes
        return "live"

    async def _show_live_mode_warning(self) -> bool:
        """
        Show live mode warning and await user confirmation.

        Returns:
            True if user confirmed to continue, False if user wants to quit
        """
        result = await self.push_screen_wait(LiveWarningModal())
        return result

    async def _switch_to_mode(self, target_mode: str) -> bool:
        """
        Switch to a new bot mode safely.

        Returns:
            True if switch was successful, False if cancelled/failed
        """
        logger.info(f"Initiating mode switch from {self.data_source_mode} to {target_mode}")

        # Step 0: Validate bot is stopped
        if self.bot.running:
            self.notify(
                "Please stop the bot before switching modes",
                severity="warning",
                title="Mode Switch",
            )
            return False

        # Step 1: For live mode, show warning modal
        if target_mode == "live":
            should_continue = await self._show_live_mode_warning()
            if not should_continue:
                logger.info("User cancelled switch to live mode")
                return False

        try:
            # Step 2: Stop any remaining bot tasks
            if self._bot_task and not self._bot_task.done():
                self._bot_task.cancel()
                try:
                    await self._bot_task
                except asyncio.CancelledError:
                    pass
                self._bot_task = None

            # Ensure bot is fully stopped
            if self.bot.running:
                await self.bot.stop()

            # Step 3: Disconnect from old bot's observer
            if hasattr(self.bot.engine, "status_reporter"):
                self.bot.engine.status_reporter.remove_observer(self._on_status_update)
                logger.info("Disconnected from old bot's status reporter")

            # Step 4: Create new bot instance
            if target_mode == "demo":
                # Create DemoBot instead of TradingBot
                from gpt_trader.tui.demo.demo_bot import DemoBot
                from gpt_trader.tui.demo.scenarios import get_scenario

                scenario = get_scenario("mixed")  # Default scenario
                new_bot = DemoBot(data_generator=scenario)
                logger.info("Created DemoBot for demo mode")
            else:
                # Create TradingBot for real modes
                from gpt_trader.cli.services import instantiate_bot

                new_config = await self._create_config_for_mode(target_mode)
                new_bot = instantiate_bot(new_config)
                logger.info(f"Created TradingBot for mode: {target_mode}")

            # Step 5: Replace bot instance
            old_bot = self.bot
            self.bot = new_bot

            # Step 6: Connect to new bot's observer
            if hasattr(self.bot.engine, "status_reporter"):
                self.bot.engine.status_reporter.add_observer(self._on_status_update)
                logger.info("Connected to new bot's status reporter")

            # Step 7: Update TUI state
            self.data_source_mode = self._detect_bot_mode()

            # Step 8: Reset UI state (fresh start)
            self.tui_state = TuiState()
            self.tui_state.data_source_mode = self.data_source_mode

            # Step 9: Sync UI immediately with new bot
            self._sync_state_from_bot()
            try:
                main_screen = self.query_one(MainScreen)
                main_screen.update_ui(self.tui_state)
                self._pulse_heartbeat()
            except Exception as e:
                logger.warning(f"Failed to update UI after mode switch: {e}")

            # Step 10: Update mode selector
            try:
                from gpt_trader.tui.widgets import ModeSelector

                mode_selector = self.query_one(ModeSelector)
                mode_selector.current_mode = self.data_source_mode
            except Exception:
                pass

            self.notify(
                f"Switched to {target_mode.upper()} mode",
                severity="information",
                title="Mode Switch",
            )
            logger.info(f"Mode switch completed successfully to {target_mode}")

            # Clean up old bot (optional, will be garbage collected)
            del old_bot

            return True

        except Exception as e:
            logger.error(f"Failed to switch modes: {e}", exc_info=True)
            self.notify(
                f"Mode switch failed: {e}",
                severity="error",
                title="Mode Switch",
            )
            return False

    async def _create_config_for_mode(self, mode: str) -> Any:
        """
        Create BotConfig for the specified mode.

        Uses predefined configs for each mode to ensure correct broker setup.
        """
        from gpt_trader.orchestration.configuration import BotConfig, Profile

        if mode == "paper":
            # Paper trading: Real data, HybridPaperBroker
            # Load from paper.yaml if exists, otherwise use defaults
            from gpt_trader.cli.services import load_config_from_yaml

            try:
                return load_config_from_yaml("config/profiles/paper.yaml")
            except Exception:
                return BotConfig.from_profile(profile=Profile.DEMO, mock_broker=False)

        elif mode == "read_only":
            # Observation mode: Real data, orders blocked
            from gpt_trader.cli.services import load_config_from_yaml

            try:
                config = load_config_from_yaml("config/profiles/observe.yaml")
                config.read_only = True
                return config
            except Exception:
                config = BotConfig.from_profile(profile=Profile.DEMO, mock_broker=False)
                config.read_only = True
                return config

        elif mode == "live":
            # Live trading: Real broker, real execution
            from gpt_trader.cli.services import load_config_from_yaml

            try:
                return load_config_from_yaml("config/profiles/prod.yaml")
            except Exception:
                return BotConfig.from_profile(profile=Profile.PROD, mock_broker=False)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle termination signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        # Schedule the exit on the event loop
        if hasattr(self, "exit"):
            self.exit()

    async def on_mount(self) -> None:
        """Called when app starts."""
        import sys

        print("[APP] TraderApp.on_mount() called", file=sys.stderr)
        try:
            # ATTACH LOG HANDLER FIRST - before any widgets mount
            print("[APP] About to attach TUI log handler", file=sys.stderr)
            from gpt_trader.tui.log_manager import attach_tui_log_handler

            attach_tui_log_handler()
            print("[APP] Attach complete, testing logger", file=sys.stderr)
            logger.info("TUI log handler attached globally")

            logger.info("TUI mounting, initializing components")

            # Mode selection flow: if no bot provided, show selection screen
            if self.bot is None:
                from gpt_trader.tui.screens.mode_selection import ModeSelectionScreen

                logger.info("No bot provided, showing mode selection screen")
                # Push the selection screen and handle selection via callback
                self.push_screen(ModeSelectionScreen(), callback=self._handle_mode_selection)
                return  # Early return - rest happens in callback

            # Direct bot provided - proceed to main screen
            await self._initialize_with_bot()

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

        # Show live warning before creating bot
        if selected_mode == "live":
            should_continue = await self._show_live_mode_warning()
            if not should_continue:
                logger.info("User declined to continue in live mode")
                self.exit()
                return

        # Create bot for selected mode
        logger.info(f"Creating bot for selected mode: {selected_mode}")
        self.bot = _create_bot_for_mode(selected_mode, self._demo_scenario)

        # Initialize with the newly created bot
        await self._initialize_with_bot()

    async def _initialize_with_bot(self) -> None:
        """Initialize the TUI with a bot instance."""
        # Detect and set bot mode
        self.data_source_mode = self._detect_bot_mode()
        self.tui_state.data_source_mode = self.data_source_mode
        logger.info(f"Bot mode detected: {self.data_source_mode}")

        # Show live mode warning if bot was provided directly and is live mode
        # (mode selection flow already showed warning before creating bot)
        if self.data_source_mode == "live" and self._initial_mode not in ("live", None):
            should_continue = await self._show_live_mode_warning()
            if not should_continue:
                logger.info("User declined to continue in live mode")
                self.exit()
                return

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

        # Initialize responsive state based on current terminal width
        self.terminal_width = self.size.width
        self.responsive_state = calculate_responsive_state(self.size.width)
        logger.info(
            f"Initial responsive state: {self.responsive_state} (width: {self.terminal_width})"
        )

    def on_resize(self, event: events.Resize) -> None:
        """Handle terminal resize events with throttling.

        Updates responsive state when terminal width changes, with 100ms
        debouncing to avoid excessive repaints during rapid resizing.
        """
        # Stop any pending resize timer
        if hasattr(self, "_resize_timer") and self._resize_timer:
            self._resize_timer.stop()

        # Schedule debounced update (100ms delay)
        self._resize_timer = self.set_timer(
            0.1, lambda: self._update_responsive_state(event.size.width)
        )

    def _update_responsive_state(self, width: int) -> None:
        """Update responsive state based on new width.

        Args:
            width: New terminal width in columns
        """
        old_state = self.responsive_state
        new_state = calculate_responsive_state(width)

        self.terminal_width = width

        if new_state != old_state:
            logger.info(f"Responsive state changed: {old_state} → {new_state} (width: {width})")
            self.responsive_state = new_state

    def watch_responsive_state(self, state: str) -> None:
        """Propagate responsive state changes to child widgets.

        Args:
            state: New responsive state
        """
        # Propagate to MainScreen if it exists
        try:
            main_screen = self.screen
            if hasattr(main_screen, "responsive_state"):
                main_screen.responsive_state = state
        except Exception:
            # Screen might not be mounted yet
            pass

    async def on_unmount(self) -> None:
        """Called when app stops - ensure all cleanup happens."""
        try:
            logger.info("TUI unmounting, cleaning up observers and tasks")

            # Detach log handler
            from gpt_trader.tui.log_manager import detach_tui_log_handler

            detach_tui_log_handler()
            logger.info("TUI log handler detached")

            # Cancel update task
            if self._update_task and not self._update_task.done():
                logger.info("Cancelling UI update task")
                self._update_task.cancel()
                try:
                    await self._update_task
                except asyncio.CancelledError:
                    logger.info("UI update task cancelled")

            # Cancel bot task if running
            if self._bot_task and not self._bot_task.done():
                logger.info("Cancelling bot task")
                self._bot_task.cancel()
                try:
                    await self._bot_task
                except asyncio.CancelledError:
                    logger.info("Bot task cancelled")

            # Stop bot if running
            if self.bot.running:
                logger.info("Stopping bot")
                await self.bot.stop()

            # Remove observer
            if hasattr(self.bot.engine, "status_reporter"):
                self.bot.engine.status_reporter.remove_observer(self._on_status_update)
                logger.info("Removed StatusReporter observer")

            logger.info("TUI unmounted successfully - all cleanup complete")
        except Exception as e:
            logger.error(f"Error during TUI unmount: {e}", exc_info=True)

    def _on_status_update(self, status: BotStatus) -> None:
        """Callback for StatusReporter updates (receives typed BotStatus)."""
        logger.debug(
            f"StatusReporter update received: bot_id={status.bot_id}, "
            f"timestamp={status.timestamp_iso}"
        )
        # This might be called from a background thread or loop
        # Schedule the update on the main thread
        self.call_from_thread(self._apply_status_update, status)

    def _apply_status_update(self, status: BotStatus) -> None:
        """Apply typed status update to state and UI."""
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
        if status.market:
            logger.debug(f"Market data: {len(status.market.last_prices)} symbols")
        if status.positions:
            logger.debug(f"Position data received: {status.positions.count} positions")

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
        """Smooth heartbeat pulse using sine wave."""
        import math
        import time

        try:
            from gpt_trader.tui.widgets.status import BotStatusWidget

            status_widget = self.query_one(BotStatusWidget)

            # Calculate sine wave: 0.0 to 1.0
            t = time.time()
            pulse = (math.sin(t * 2) + 1) / 2  # Sine wave normalized to 0-1

            status_widget.heartbeat = pulse
        except Exception as e:
            logger.debug(f"Failed to pulse heartbeat: {e}")

    async def _update_loop(self) -> None:
        """Periodically update UI from bot state (Fallback + heartbeat)."""
        loop_count = 0
        while True:
            try:
                loop_count += 1

                # ALWAYS poll and update - provides fallback + heartbeat
                if loop_count % 10 == 0:  # Log occasionally for debugging
                    observer_count = (
                        len(self.bot.engine.status_reporter._observers)
                        if hasattr(self.bot.engine, "status_reporter")
                        else 0
                    )
                    logger.debug(
                        f"Update loop tick {loop_count}, "
                        f"observers: {observer_count}, "
                        f"bot running: {self.bot.running}"
                    )

                # Sync state (fast if no changes)
                self._sync_state_from_bot()

                # Update UI
                try:
                    main_screen = self.query_one(MainScreen)
                    main_screen.update_ui(self.tui_state)
                    # Pulse heartbeat to show dashboard is alive
                    self._pulse_heartbeat()
                except Exception as e:
                    logger.warning(f"Failed to update main screen: {e}")

            except Exception as e:
                self.log(f"UI Update Error: {e}")
                logger.error(f"UI Update Loop Error: {e}", exc_info=True)

            await asyncio.sleep(1)

    def _sync_state_from_bot(self) -> None:
        """Fetch typed state from bot and update TuiState."""
        self.tui_state.running = self.bot.running

        # Access runtime state safely
        runtime_state = None
        if hasattr(self.bot.engine.context, "runtime_state"):
            runtime_state = self.bot.engine.context.runtime_state

        # Access StatusReporter for typed data
        if hasattr(self.bot.engine, "status_reporter"):
            status = self.bot.engine.status_reporter.get_status()  # Returns BotStatus
            logger.debug(
                f"Fetched status from StatusReporter: bot_id={status.bot_id}, "
                f"timestamp={status.timestamp_iso}"
            )
            self.tui_state.update_from_bot_status(status, runtime_state)
        else:
            logger.debug("No StatusReporter available, skipping status update")

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

                # Re-enable mode selector after stop
                try:
                    from gpt_trader.tui.widgets import ModeSelector

                    mode_selector = self.query_one(ModeSelector)
                    mode_selector.enabled = True
                except Exception:
                    pass

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
                # Disable mode selector before start
                try:
                    from gpt_trader.tui.widgets import ModeSelector

                    mode_selector = self.query_one(ModeSelector)
                    mode_selector.enabled = False
                except Exception:
                    pass

                # Start the bot and store task reference
                self.notify("Starting bot...", title="Status")
                logger.info("=" * 60)
                logger.info("BOT STARTING - User initiated bot start via TUI")
                logger.info(f"Bot type: {type(self.bot).__name__}")
                logger.info(f"Data source mode: {self.data_source_mode}")
                logger.info("=" * 60)

                # Store task reference for future cancellation
                self._bot_task = asyncio.create_task(self.bot.run())

                # Add a done callback to log if the task fails
                def _log_bot_task_done(task: asyncio.Task) -> None:
                    try:
                        if task.cancelled():
                            logger.warning("Bot task was cancelled")
                        elif task.exception():
                            logger.error(
                                f"Bot task failed with exception: {task.exception()}",
                                exc_info=task.exception(),
                            )
                        else:
                            logger.info("Bot task completed normally")
                    except Exception as e:
                        logger.error(f"Error in bot task done callback: {e}")

                self._bot_task.add_done_callback(_log_bot_task_done)

                # Give it a moment to start
                await asyncio.sleep(0.2)

                # Check if task is still running
                if self._bot_task.done():
                    logger.error("Bot task completed immediately! This is unexpected.")
                    try:
                        # Try to get the exception if there was one
                        self._bot_task.result()
                    except Exception as e:
                        logger.error(f"Bot task failed on startup: {e}", exc_info=True)
                else:
                    logger.info("Bot task is running successfully")

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
        try:
            # Determine which screen we're on and query appropriate widget
            if isinstance(self.screen, MainScreen):
                log_widget = self.query_one("#dash-logs")
            elif isinstance(self.screen, FullLogsScreen):
                log_widget = self.query_one("#full-logs")
            else:
                # Other screens may not have log widgets
                self.notify("No log widget on this screen", severity="information")
                return

            log_widget.focus()
        except Exception as e:
            logger.warning(f"Failed to focus log widget: {e}")
            self.notify("Could not focus logs widget", severity="warning")

    async def action_show_full_logs(self) -> None:
        """Show full logs screen (expanded view)."""
        try:
            logger.debug("Opening full logs screen")
            self.push_screen(FullLogsScreen())
        except Exception as e:
            logger.error(f"Failed to show full logs screen: {e}", exc_info=True)
            self.notify(f"Error showing full logs: {e}", severity="error")

    async def action_show_system_details(self) -> None:
        """Show detailed system metrics screen."""
        try:
            logger.debug("Opening system details screen")
            self.push_screen(SystemDetailsScreen())
        except Exception as e:
            logger.error(f"Failed to show system details screen: {e}", exc_info=True)
            self.notify(f"Error showing system details: {e}", severity="error")

    async def action_show_mode_info(self) -> None:
        """Show detailed mode information modal."""
        try:
            logger.debug(f"Opening mode info modal for mode: {self.data_source_mode}")
            self.push_screen(ModeInfoModal(self.data_source_mode))
        except Exception as e:
            logger.error(f"Failed to show mode info modal: {e}", exc_info=True)
            self.notify(f"Error showing mode info: {e}", severity="error")

    async def action_reconnect_data(self) -> None:
        """Attempt to reconnect data source."""
        try:
            if self.data_source_mode == "demo":
                self.notify("Demo mode doesn't require reconnection", severity="information")
                return

            logger.info("User initiated data source reconnection")
            self.notify("Reconnecting to Coinbase...", title="Connection")

            # Trigger a status sync
            self._sync_state_from_bot()
            await asyncio.sleep(0.5)

            # Check connection health
            is_healthy = self.tui_state.check_connection_health()
            if is_healthy:
                self.notify("Reconnected successfully", title="Connection", severity="information")
            else:
                self.notify(
                    "Connection may still be stale, please wait...",
                    title="Connection",
                    severity="warning",
                )
        except Exception as e:
            logger.error(f"Failed to reconnect: {e}", exc_info=True)
            self.notify(f"Error reconnecting: {e}", severity="error")

    def on_bot_status_widget_toggle_bot_pressed(
        self, message: BotStatusWidget.ToggleBotPressed
    ) -> None:
        """Handle start/stop button press from BotStatusWidget."""
        asyncio.create_task(self.action_toggle_bot())

    def on_mode_selector_mode_changed(self, message: Any) -> None:
        """Handle mode change request from ModeSelector."""
        from gpt_trader.tui.widgets import ModeSelector

        if isinstance(message, ModeSelector.ModeChanged):
            asyncio.create_task(self._switch_to_mode(message.new_mode))

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

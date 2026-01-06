"""
Bot Lifecycle Manager for TUI.

Handles bot creation, starting, stopping, and mode switching operations.
Extracted from TraderApp to reduce complexity and improve testability.

Supports two modes of operation:
1. Worker-based (preferred): Uses Textual Workers for better lifecycle management
2. Legacy task-based: Uses raw asyncio.Task for backward compatibility
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from textual.worker import Worker, WorkerState

from gpt_trader.tui.notification_helpers import notify_error, notify_success, notify_warning
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp
    from gpt_trader.tui.services.worker_service import WorkerService

logger = get_logger(__name__, component="tui")


class BotLifecycleManager:
    """Manages bot creation, lifecycle operations, and mode switching.

    Supports both Worker-based and legacy task-based bot execution.
    Worker mode is preferred for better lifecycle management.
    """

    def __init__(
        self,
        app: TraderApp,
        worker_service: WorkerService | None = None,
    ):
        """
        Initialize BotLifecycleManager.

        Args:
            app: Reference to the TraderApp instance
            worker_service: Optional WorkerService for Worker-based execution
        """
        self.app = app
        self.worker_service = worker_service
        self._bot_task: asyncio.Task | None = None
        self._bot_worker: Worker[None] | None = None
        self._use_workers = worker_service is not None

    def detect_bot_mode(self) -> str:
        """
        Detect current bot operating mode.

        Returns:
            "demo" - Mock data simulation
            "paper" - Real data with simulated execution
            "read_only" - Real data observation, no execution
            "live" - Real trading with real execution
        """
        # Check bot type first
        if type(self.app.bot).__name__ == "DemoBot":
            return "demo"

        # Check for read-only mode in config
        if hasattr(self.app.bot, "config") and getattr(self.app.bot.config, "read_only", False):
            return "read_only"

        # Check broker type
        if hasattr(self.app.bot, "engine") and hasattr(self.app.bot.engine, "broker"):
            broker_type = type(self.app.bot.engine.broker).__name__

            if "Paper" in broker_type or "Mock" in broker_type:
                return "paper"
            return "live"

        # Default fallback
        return "demo"

    def _is_bot_running_internally(self) -> bool:
        """Check if bot is running based on internal task/worker state."""
        if self._use_workers and self._bot_worker:
            return self._bot_worker.state == WorkerState.RUNNING
        return self._bot_task is not None and not self._bot_task.done()

    async def toggle_bot(self) -> None:
        """Toggle bot running state (start/stop)."""
        try:
            if self.app.bot.running and self._is_bot_running_internally():
                await self.stop_bot()
            elif not self.app.bot.running:
                await self.start_bot()
            else:
                # Edge case: bot says it's running but we don't have task/worker reference
                logger.warning("Bot state inconsistent: running=True but no task/worker reference")
                notify_warning(self.app, "Bot state inconsistent, attempting recovery...")
                await self.app.bot.stop()
                self._bot_task = None
                self._bot_worker = None

        except Exception as e:
            logger.error(f"Failed to toggle bot state: {e}", exc_info=True)
            notify_error(self.app, f"Failed to toggle bot: {e}", title="Bot Control")

    async def start_bot(self) -> None:
        """Start the bot.

        Uses Worker-based execution if WorkerService is available,
        otherwise falls back to legacy asyncio.Task.
        """
        # Disable mode selector before start
        self._set_mode_selector_enabled(False)

        # Log start (notification will show after bot actually starts)
        logger.info("=" * 60)
        logger.info("BOT STARTING - User initiated bot start via TUI")
        logger.info(f"Bot type: {type(self.app.bot).__name__}")
        logger.info(f"Data source mode: {self.app.data_source_mode}")
        logger.info(f"Execution mode: {'Worker' if self._use_workers else 'Task'}")
        logger.info("=" * 60)

        # Reset trade matcher on bot start via event (decoupled)
        from gpt_trader.tui.events import TradeMatcherResetRequested

        self.app.post_message(TradeMatcherResetRequested())
        logger.info("Posted TradeMatcherResetRequested for fresh P&L tracking on bot start")

        if self._use_workers and self.worker_service:
            # Worker-based execution (preferred)
            self._bot_worker = self.worker_service.run_bot_async()
            logger.info(f"Bot started via Worker: {self._bot_worker.name}")

            # Check worker state immediately (no artificial delay)
            # Worker state checking is non-blocking
            if self._bot_worker.state == WorkerState.ERROR:
                logger.error("Bot worker failed on startup")
                if self._bot_worker.error:
                    logger.error(f"Worker error: {self._bot_worker.error}")
            elif self._bot_worker.state in (WorkerState.RUNNING, WorkerState.PENDING):
                logger.info("Bot worker started successfully")
        else:
            # Legacy task-based execution
            self._bot_task = asyncio.create_task(self.app.bot.run())

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

            # Check task state immediately (no artificial delay)
            # Task done() check is non-blocking
            if self._bot_task.done():
                logger.error("Bot task completed immediately! This is unexpected.")
                try:
                    self._bot_task.result()
                except Exception as e:
                    logger.error(f"Bot task failed on startup: {e}", exc_info=True)
            else:
                logger.info("Bot task started successfully")

        # Immediately sync state to UI
        self.app._sync_state_from_bot()
        self._update_main_screen()

        notify_success(self.app, "Bot started.", title="Status")
        logger.info("Bot started successfully via TUI")

    async def stop_bot(self) -> None:
        """Stop the bot.

        Handles both Worker-based and legacy task-based execution.
        """
        logger.info("User initiated bot stop via TUI")

        if self._use_workers and self._bot_worker:
            # Worker-based cancellation
            if self._bot_worker.state == WorkerState.RUNNING:
                self._bot_worker.cancel()
                logger.info("Bot worker cancellation requested")
                # Cancellation is async - no need to wait
            self._bot_worker = None
        elif self._bot_task:
            # Legacy task-based cancellation
            self._bot_task.cancel()
            try:
                await self._bot_task
            except asyncio.CancelledError:
                logger.info("Bot task cancelled successfully")
            self._bot_task = None

        # Ensure bot.stop() is called for cleanup
        await self.app.bot.stop()

        # Re-enable mode selector after stop
        self._set_mode_selector_enabled(True)

        # Immediately sync state to UI
        self.app._sync_state_from_bot()
        self._update_main_screen()

        notify_success(self.app, "Bot stopped.", title="Status")
        logger.info("Bot stopped successfully via TUI")

    def panic_stop(self) -> None:
        """
        Emergency panic stop: stop bot and flatten all positions.

        This is a critical safety mechanism. Errors are logged but
        do not prevent the stop sequence from completing.
        """
        try:
            # 1. Stop bot immediately
            if self.app.bot and hasattr(self.app.bot, "running") and self.app.bot.running:
                logger.warning("Panic stop: Stopping bot")
                # Use asyncio to run the async stop_bot method
                import asyncio

                try:
                    asyncio.create_task(self.stop_bot())
                except RuntimeError:
                    # If event loop not running, try synchronous stop
                    if hasattr(self.app.bot, "stop"):
                        self.app.bot.stop()

            # 2. Flatten all positions (if broker supports it)
            logger.warning("Panic stop: Attempting to flatten all positions")
            if self.app.bot and hasattr(self.app.bot, "flatten_all_positions"):
                try:
                    self.app.bot.flatten_all_positions()
                    logger.info("Panic stop: Position flattening initiated")
                except Exception as e:
                    logger.error(f"Panic stop: Position flattening failed: {e}", exc_info=True)
            else:
                logger.warning("Bot does not support flatten_all_positions()")

        except Exception as e:
            logger.error(f"Panic stop encountered error: {e}", exc_info=True)
            # Don't re-raise - panic must complete even if errors occur

    async def switch_mode(self, target_mode: str) -> bool:
        """
        Switch to a new bot mode safely.

        Args:
            target_mode: Target mode (demo, paper, read_only, live)

        Returns:
            True if switch was successful, False if cancelled/failed
        """
        logger.info(f"Initiating mode switch from {self.app.data_source_mode} to {target_mode}")

        # Step 0: Validate bot is stopped
        if self.app.bot.running:
            notify_warning(
                self.app, "Please stop the bot before switching modes", title="Mode Switch"
            )
            return False

        # Step 1: For live mode, show warning modal
        if target_mode == "live":
            should_continue = await self._show_live_mode_warning()
            if not should_continue:
                logger.info("User cancelled switch to live mode")
                return False

        # Show loading indicator
        self._set_mode_selector_loading(True)

        try:
            # Step 2: Stop any remaining bot tasks/workers
            if self._use_workers and self._bot_worker:
                if self._bot_worker.state == WorkerState.RUNNING:
                    self._bot_worker.cancel()
                self._bot_worker = None
            elif self._bot_task and not self._bot_task.done():
                self._bot_task.cancel()
                try:
                    await self._bot_task
                except asyncio.CancelledError:
                    pass
                self._bot_task = None

            # Ensure bot is fully stopped
            if self.app.bot.running:
                await self.app.bot.stop()

            # Step 3: Disconnect from old bot's observer (skip for NullStatusReporter)
            if hasattr(self.app.bot.engine, "status_reporter"):
                reporter = self.app.bot.engine.status_reporter
                if not getattr(reporter, "is_null_reporter", False):
                    reporter.remove_observer(self.app._on_status_update)
                    logger.info("Disconnected from old bot's status reporter")
                else:
                    logger.debug("Skipping observer disconnect for NullStatusReporter")

            # Step 4: Create new bot instance
            if target_mode == "demo":
                new_bot = await self._create_demo_bot()
            else:
                new_bot = await self._create_trading_bot(target_mode)

            # Step 5: Replace bot instance
            old_bot = self.app.bot
            self.app.bot = new_bot

            # Step 6: Connect to new bot's observer (skip for NullStatusReporter)
            if hasattr(self.app.bot.engine, "status_reporter"):
                reporter = self.app.bot.engine.status_reporter
                if not getattr(reporter, "is_null_reporter", False):
                    reporter.add_observer(self.app._on_status_update)
                    logger.info("Connected to new bot's status reporter")
                    # Exit degraded mode if we were in it
                    self.app.tui_state.degraded_mode = False
                    self.app.tui_state.degraded_reason = ""
                else:
                    logger.info("NullStatusReporter on new bot - entering degraded mode")
                    self.app.tui_state.degraded_mode = True
                    self.app.tui_state.degraded_reason = "StatusReporter not available"
                    self.app.tui_state.connection_healthy = False

            # Step 7: Update TUI state
            self.app.data_source_mode = self.detect_bot_mode()

            # Step 8: Reset UI state (fresh start)
            from gpt_trader.tui.state import TuiState

            self.app.tui_state = TuiState()
            self.app.tui_state.data_source_mode = self.app.data_source_mode

            # Step 8.5: Reset trade matcher via event (decoupled)
            # Clear P&L tracking state when switching modes
            from gpt_trader.tui.events import TradeMatcherResetRequested

            self.app.post_message(TradeMatcherResetRequested())
            logger.info("Posted TradeMatcherResetRequested for fresh P&L tracking on mode switch")

            # Step 9: Sync UI immediately with new bot
            self.app._sync_state_from_bot()
            self._update_main_screen()

            # Step 10: Update mode selector
            self._update_mode_selector(self.app.data_source_mode)

            # Hide loading indicator on success
            self._set_mode_selector_loading(False)

            # Save mode preference for future launches
            self.app.mode_service.save_mode_preference(target_mode)

            notify_success(self.app, f"Switched to {target_mode.upper()} mode", title="Mode Switch")
            logger.info(f"Mode switch completed successfully to {target_mode}")

            # Clean up old bot (optional, will be garbage collected)
            del old_bot

            return True

        except Exception as e:
            # Hide loading indicator on failure
            self._set_mode_selector_loading(False)

            logger.error(f"Failed to switch modes: {e}", exc_info=True)
            notify_error(self.app, f"Mode switch failed: {e}", title="Mode Switch")
            return False

    async def _create_demo_bot(self) -> Any:
        """Create a DemoBot instance."""
        from gpt_trader.tui.demo.demo_bot import DemoBot
        from gpt_trader.tui.demo.scenarios import get_scenario

        scenario = get_scenario("mixed")  # Default scenario
        bot = DemoBot(data_generator=scenario)
        logger.info("Created DemoBot for demo mode")
        return bot

    async def _create_trading_bot(self, mode: str) -> Any:
        """
        Create a TradingBot instance for the specified mode.

        Args:
            mode: Mode to create bot for (paper, read_only, live)

        Returns:
            TradingBot instance
        """
        from gpt_trader.cli.services import instantiate_bot

        config = await self._create_config_for_mode(mode)
        bot = instantiate_bot(config)
        logger.info(f"Created TradingBot for mode: {mode}")
        return bot

    async def _create_config_for_mode(self, mode: str) -> Any:
        """
        Create BotConfig for the specified mode.

        Args:
            mode: Mode to create config for (paper, read_only, live)

        Returns:
            BotConfig instance

        Raises:
            ValueError: If mode is unknown
        """
        from gpt_trader.app.config import BotConfig
        from gpt_trader.config.types import Profile

        if mode == "paper":
            # Paper trading: Real data, HybridPaperBroker
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

    async def _show_live_mode_warning(self) -> bool:
        """
        Show live mode warning modal and get user confirmation.

        Returns:
            True if user confirmed, False if cancelled
        """
        from gpt_trader.tui.widgets import LiveWarningModal

        result = await self.app.push_screen_wait(LiveWarningModal())
        return bool(result)

    def _set_mode_selector_enabled(self, enabled: bool) -> None:
        """Enable or disable the mode selector widget via event."""
        from gpt_trader.tui.events import ModeSelectorEnabledChanged

        self.app.post_message(ModeSelectorEnabledChanged(enabled=enabled))
        logger.debug(f"Posted ModeSelectorEnabledChanged(enabled={enabled})")

    def _update_mode_selector(self, mode: str) -> None:
        """Update the mode selector widget via event."""
        from gpt_trader.tui.events import ModeSelectorValueChanged

        self.app.post_message(ModeSelectorValueChanged(mode=mode))
        logger.debug(f"Posted ModeSelectorValueChanged(mode={mode})")

    def _set_mode_selector_loading(self, loading: bool) -> None:
        """Show or hide loading indicator on mode selector."""
        from gpt_trader.tui.events import ModeSelectorLoadingChanged

        self.app.post_message(ModeSelectorLoadingChanged(loading=loading))
        logger.debug(f"Posted ModeSelectorLoadingChanged(loading={loading})")

    def _update_main_screen(self) -> None:
        """Update the main screen UI via event."""
        from gpt_trader.tui.events import MainScreenRefreshRequested

        self.app.post_message(MainScreenRefreshRequested())
        logger.info("Posted MainScreenRefreshRequested for UI sync")

    def cleanup(self) -> None:
        """Clean up bot tasks/workers on manager destruction."""
        if self._use_workers and self._bot_worker:
            if self._bot_worker.state == WorkerState.RUNNING:
                self._bot_worker.cancel()
            self._bot_worker = None
            logger.info("BotLifecycleManager cleaned up bot worker")
        elif self._bot_task and not self._bot_task.done():
            self._bot_task.cancel()
            self._bot_task = None
            logger.info("BotLifecycleManager cleaned up bot task")

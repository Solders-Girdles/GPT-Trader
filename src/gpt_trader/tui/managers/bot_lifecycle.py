"""
Bot Lifecycle Manager for TUI.

Handles bot creation, starting, stopping, and mode switching operations.
Extracted from TraderApp to reduce complexity and improve testability.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp

logger = get_logger(__name__, component="tui")


class BotLifecycleManager:
    """Manages bot creation, lifecycle operations, and mode switching."""

    def __init__(self, app: TraderApp):
        """
        Initialize BotLifecycleManager.

        Args:
            app: Reference to the TraderApp instance
        """
        self.app = app
        self._bot_task: asyncio.Task | None = None

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

    async def toggle_bot(self) -> None:
        """Toggle bot running state (start/stop)."""
        try:
            if self.app.bot.running and self._bot_task:
                await self.stop_bot()
            elif not self.app.bot.running:
                await self.start_bot()
            else:
                # Edge case: bot says it's running but we don't have task reference
                logger.warning("Bot state inconsistent: running=True but no task reference")
                self.app.notify(
                    "Bot state inconsistent, attempting recovery...", severity="warning"
                )
                await self.app.bot.stop()
                self._bot_task = None

        except Exception as e:
            logger.error(f"Failed to toggle bot state: {e}", exc_info=True)
            self.app.notify(f"Failed to toggle bot: {e}", severity="error", title="Bot Control")

    async def start_bot(self) -> None:
        """Start the bot."""
        # Disable mode selector before start
        self._set_mode_selector_enabled(False)

        # Notify start
        self.app.notify("Starting bot...", title="Status")
        logger.info("=" * 60)
        logger.info("BOT STARTING - User initiated bot start via TUI")
        logger.info(f"Bot type: {type(self.app.bot).__name__}")
        logger.info(f"Data source mode: {self.app.data_source_mode}")
        logger.info("=" * 60)

        # Reset trade matcher on bot start (Phase 6 - Incremental P&L)
        try:
            from gpt_trader.tui.screens.main import MainScreen
            from gpt_trader.tui.widgets.positions import TradesWidget

            main_screen = self.app.query_one(MainScreen)
            exec_widget = main_screen.query_one("#dash-execution")
            trades_widget = exec_widget.query_one(TradesWidget)
            if hasattr(trades_widget, "_trade_matcher"):
                trades_widget._trade_matcher.reset()
                logger.info("Reset trade matcher for fresh P&L tracking on bot start")
        except Exception as e:
            logger.debug(f"Could not reset trade matcher on start: {e}")

        # Store task reference for future cancellation
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
        self.app._sync_state_from_bot()
        self._update_main_screen()

        self.app.notify("Bot started.", title="Status", severity="information")
        logger.info("Bot started successfully via TUI")

    async def stop_bot(self) -> None:
        """Stop the bot."""
        # Stop the bot with proper task cancellation
        self.app.notify("Stopping bot...", title="Status")
        logger.info("User initiated bot stop via TUI")

        # Cancel the running task
        if self._bot_task:
            self._bot_task.cancel()
            try:
                await self._bot_task
            except asyncio.CancelledError:
                logger.info("Bot task cancelled successfully")

            # Clean up task reference
            self._bot_task = None

        # Ensure bot.stop() is called for cleanup
        await self.app.bot.stop()

        # Re-enable mode selector after stop
        self._set_mode_selector_enabled(True)

        # Immediately sync state to UI
        self.app._sync_state_from_bot()
        self._update_main_screen()

        self.app.notify("Bot stopped.", title="Status", severity="information")
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
            self.app.notify(
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
            if self.app.bot.running:
                await self.app.bot.stop()

            # Step 3: Disconnect from old bot's observer
            if hasattr(self.app.bot.engine, "status_reporter"):
                self.app.bot.engine.status_reporter.remove_observer(self.app._on_status_update)
                logger.info("Disconnected from old bot's status reporter")

            # Step 4: Create new bot instance
            if target_mode == "demo":
                new_bot = await self._create_demo_bot()
            else:
                new_bot = await self._create_trading_bot(target_mode)

            # Step 5: Replace bot instance
            old_bot = self.app.bot
            self.app.bot = new_bot

            # Step 6: Connect to new bot's observer
            if hasattr(self.app.bot.engine, "status_reporter"):
                self.app.bot.engine.status_reporter.add_observer(self.app._on_status_update)
                logger.info("Connected to new bot's status reporter")

            # Step 7: Update TUI state
            self.app.data_source_mode = self.detect_bot_mode()

            # Step 8: Reset UI state (fresh start)
            from gpt_trader.tui.state import TuiState

            self.app.tui_state = TuiState()
            self.app.tui_state.data_source_mode = self.app.data_source_mode

            # Step 8.5: Reset trade matcher (Phase 6 - Incremental P&L)
            # Clear P&L tracking state when switching modes
            try:
                from gpt_trader.tui.screens.main import MainScreen
                from gpt_trader.tui.widgets.positions import TradesWidget

                main_screen = self.app.query_one(MainScreen)
                exec_widget = main_screen.query_one("#dash-execution")
                trades_widget = exec_widget.query_one(TradesWidget)
                if hasattr(trades_widget, "_trade_matcher"):
                    trades_widget._trade_matcher.reset()
                    logger.info("Reset trade matcher for fresh P&L tracking")
            except Exception as e:
                logger.debug(f"Could not reset trade matcher: {e}")

            # Step 9: Sync UI immediately with new bot
            self.app._sync_state_from_bot()
            self._update_main_screen()

            # Step 10: Update mode selector
            self._update_mode_selector(self.app.data_source_mode)

            self.app.notify(
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
            self.app.notify(
                f"Mode switch failed: {e}",
                severity="error",
                title="Mode Switch",
            )
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
        from gpt_trader.orchestration.configuration import BotConfig, Profile

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
        """Enable or disable the mode selector widget."""
        try:
            from gpt_trader.tui.widgets import ModeSelector

            mode_selector = self.app.query_one(ModeSelector)
            mode_selector.enabled = enabled
        except Exception:
            # Mode selector might not be mounted yet
            pass

    def _update_mode_selector(self, mode: str) -> None:
        """Update the mode selector widget to reflect current mode."""
        try:
            from gpt_trader.tui.widgets import ModeSelector

            mode_selector = self.app.query_one(ModeSelector)
            mode_selector.current_mode = mode
        except Exception:
            # Mode selector might not be mounted yet
            pass

    def _update_main_screen(self) -> None:
        """Update the main screen UI."""
        try:
            from gpt_trader.tui.screens import MainScreen

            main_screen = self.app.query_one(MainScreen)
            main_screen.update_ui(self.app.tui_state)
            self.app._pulse_heartbeat()
            logger.info("UI state synced after bot lifecycle operation")
        except Exception as e:
            logger.warning(f"Failed to update main screen: {e}")

    def cleanup(self) -> None:
        """Clean up bot tasks on manager destruction."""
        if self._bot_task and not self._bot_task.done():
            self._bot_task.cancel()
            logger.info("BotLifecycleManager cleaned up bot task")

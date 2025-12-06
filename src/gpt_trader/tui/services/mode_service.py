"""
Mode service for managing bot mode creation and switching.

Handles the creation of bot instances for different operating modes
(demo, paper, read_only, live) and mode switching logic. Also manages
mode persistence to remember the user's last selected mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpt_trader.tui.events import BotModeChanged
from gpt_trader.tui.widgets import LiveWarningModal
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from textual.app import App

    from gpt_trader.orchestration.trading_bot.bot import TradingBot

logger = get_logger(__name__, component="tui")

# Default config file path for mode preferences (shared with theme)
DEFAULT_PREFERENCES_PATH = Path("config/tui_preferences.json")


def create_bot_for_mode(mode: str, demo_scenario: str = "mixed") -> Any:
    """Create a bot instance for the specified mode.

    Args:
        mode: One of "demo", "paper", "read_only", "live"
        demo_scenario: Scenario to use for demo mode

    Returns:
        Bot instance (DemoBot or TradingBot)

    Raises:
        ValueError: If mode is unknown.
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
                config.read_only = True  # type: ignore[attr-defined]
            except Exception:
                config = BotConfig.from_profile(profile=Profile.DEMO, mock_broker=False)
                config.read_only = True  # type: ignore[attr-defined]
        elif mode == "live":
            try:
                config = load_config_from_yaml("config/profiles/prod.yaml")
            except Exception:
                config = BotConfig.from_profile(profile=Profile.PROD, mock_broker=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return instantiate_bot(config)


class ModeService:
    """Service for managing bot operating modes.

    Handles mode detection, creation of bots for different modes,
    mode switching operations, and mode preference persistence.

    Attributes:
        app: Reference to the parent Textual app.
        demo_scenario: Default scenario for demo mode.
        preferences_path: Path to the preferences file.
    """

    # Valid modes for persistence
    VALID_MODES = {"demo", "paper", "read_only", "live"}

    def __init__(
        self,
        app: App,
        demo_scenario: str = "mixed",
        preferences_path: Path | None = None,
    ) -> None:
        """Initialize the mode service.

        Args:
            app: The parent Textual app.
            demo_scenario: Default scenario for demo mode.
            preferences_path: Optional custom path for preferences file.
        """
        self.app = app
        self.demo_scenario = demo_scenario
        self.preferences_path = preferences_path or DEFAULT_PREFERENCES_PATH

    def load_mode_preference(self) -> str | None:
        """Load mode preference from config file.

        Returns:
            The saved mode string, or None if not found or invalid.
        """
        try:
            if self.preferences_path.exists():
                with open(self.preferences_path) as f:
                    prefs = json.load(f)
                    mode = prefs.get("mode")
                    if mode in self.VALID_MODES:
                        logger.info(f"Loaded mode preference: {mode}")
                        return mode
                    elif mode is not None:
                        logger.debug(f"Invalid saved mode '{mode}', ignoring")
        except Exception as e:
            logger.debug(f"Could not load mode preference: {e}")

        return None

    def save_mode_preference(self, mode: str) -> bool:
        """Save mode preference to config file.

        Preserves other keys in the preferences file (e.g., theme).

        Args:
            mode: The mode to save (demo, paper, read_only, live).

        Returns:
            True if saved successfully, False otherwise.
        """
        if mode not in self.VALID_MODES:
            logger.warning(f"Cannot save invalid mode: {mode}")
            return False

        try:
            self.preferences_path.parent.mkdir(parents=True, exist_ok=True)

            prefs = {}
            if self.preferences_path.exists():
                with open(self.preferences_path) as f:
                    prefs = json.load(f)

            prefs["mode"] = mode

            with open(self.preferences_path, "w") as f:
                json.dump(prefs, f, indent=2)

            logger.info(f"Saved mode preference: {mode}")
            return True
        except Exception as e:
            logger.warning(f"Could not save mode preference: {e}")
            return False

    def detect_bot_mode(self, bot: TradingBot | Any) -> str:
        """Detect the current bot operating mode.

        Args:
            bot: The bot instance to check.

        Returns:
            Mode string: "demo", "paper", "read_only", or "live"
        """
        # Check for DemoBot
        bot_class_name = bot.__class__.__name__
        if "DemoBot" in bot_class_name or "Demo" in bot_class_name:
            return "demo"

        # Check config for mode indicators
        if hasattr(bot, "config"):
            config = bot.config

            # Check read_only flag
            if hasattr(config, "read_only") and config.read_only:
                return "read_only"

            # Check profile
            if hasattr(config, "profile"):
                profile = str(config.profile).lower()
                if "prod" in profile:
                    return "live"
                if "paper" in profile or "demo" in profile:
                    return "paper"

            # Check mock_broker flag
            if hasattr(config, "mock_broker") and config.mock_broker:
                return "paper"

        # Default to paper for safety
        return "paper"

    def create_bot(self, mode: str) -> Any:
        """Create a bot instance for the specified mode.

        Args:
            mode: One of "demo", "paper", "read_only", "live"

        Returns:
            Bot instance (DemoBot or TradingBot)
        """
        return create_bot_for_mode(mode, self.demo_scenario)

    async def show_live_warning(self) -> bool:
        """Show the live mode warning modal.

        Returns:
            True if user confirms to continue, False otherwise.
        """
        result = await self.app.push_screen_wait(LiveWarningModal())
        return bool(result)

    async def handle_mode_selection(
        self,
        selected_mode: str | None,
        on_bot_created: Any = None,
    ) -> TradingBot | Any | None:
        """Handle mode selection from the mode selection screen.

        Args:
            selected_mode: The selected mode, or None if cancelled.
            on_bot_created: Optional callback when bot is created.

        Returns:
            The created bot instance, or None if cancelled or failed.
        """
        if selected_mode is None:
            logger.info("User cancelled mode selection")
            return None

        # Show live warning before creating bot
        if selected_mode == "live":
            should_continue = await self.show_live_warning()
            if not should_continue:
                logger.info("User declined to continue in live mode")
                return None

        # Create bot for selected mode
        logger.info(f"Creating bot for selected mode: {selected_mode}")
        bot = self.create_bot(selected_mode)

        if on_bot_created:
            on_bot_created(bot)

        return bot

    def notify_mode_changed(self, new_mode: str, old_mode: str) -> None:
        """Notify that bot mode has changed.

        Posts a BotModeChanged event for handlers to process.

        Args:
            new_mode: The new mode.
            old_mode: The previous mode.
        """
        self.app.post_message(BotModeChanged(new_mode=new_mode, old_mode=old_mode))
        logger.info(f"Mode changed: {old_mode} -> {new_mode}")

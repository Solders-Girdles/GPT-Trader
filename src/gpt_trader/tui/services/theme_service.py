"""
Theme service for managing TUI themes.

Handles theme loading, persistence, and toggling between
dark and light modes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from gpt_trader.tui.events import ThemeChanged
from gpt_trader.tui.theme import Theme, ThemeMode, get_theme_manager
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from textual.app import App

logger = get_logger(__name__, component="tui")

# Default config file path for theme preferences
DEFAULT_PREFERENCES_PATH = Path("config/tui_preferences.json")


class ThemeService:
    """Service for managing TUI theme preferences.

    Handles loading and saving theme preferences to a config file,
    and provides methods for toggling between themes.

    Attributes:
        app: Reference to the parent Textual app.
        theme_manager: The global theme manager instance.
        preferences_path: Path to the preferences file.
    """

    def __init__(
        self,
        app: App,
        preferences_path: Path | None = None,
    ) -> None:
        """Initialize the theme service.

        Args:
            app: The parent Textual app.
            preferences_path: Optional custom path for preferences file.
        """
        self.app = app
        self.theme_manager = get_theme_manager()
        self.preferences_path = preferences_path or DEFAULT_PREFERENCES_PATH

    def load_preference(self) -> ThemeMode:
        """Load theme preference from config file.

        Returns:
            The loaded theme mode, or DARK as default.
        """
        try:
            if self.preferences_path.exists():
                with open(self.preferences_path) as f:
                    prefs = json.load(f)
                    mode_str = prefs.get("theme", "dark")
                    mode = ThemeMode(mode_str)
                    self.theme_manager.set_theme(mode)
                    logger.info(f"Loaded theme preference: {mode_str}")
                    return mode
        except Exception as e:
            logger.debug(f"Could not load theme preference: {e}")

        return ThemeMode.DARK

    def save_preference(self, mode: ThemeMode) -> bool:
        """Save theme preference to config file.

        Args:
            mode: The theme mode to save.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            self.preferences_path.parent.mkdir(parents=True, exist_ok=True)

            prefs = {}
            if self.preferences_path.exists():
                with open(self.preferences_path) as f:
                    prefs = json.load(f)

            prefs["theme"] = mode.value

            with open(self.preferences_path, "w") as f:
                json.dump(prefs, f, indent=2)

            logger.info(f"Saved theme preference: {mode}")
            return True
        except Exception as e:
            logger.warning(f"Could not save theme preference: {e}")
            return False

    def toggle_theme(self) -> Theme:
        """Toggle between dark and light themes.

        Returns:
            The new theme after toggling.
        """
        new_theme = self.theme_manager.toggle_theme()
        mode_name = "Light" if new_theme.mode == ThemeMode.LIGHT else "Dark"

        # Notify user
        self.app.notify(
            f"Switched to {mode_name} theme (restart to apply)",
            severity="information",
            title="Theme",
        )

        # Save preference
        self.save_preference(new_theme.mode)

        # Post event for interested widgets
        self.app.post_message(ThemeChanged(theme_mode=new_theme.mode.value))

        logger.info(f"Theme toggled to: {new_theme.mode}")
        return new_theme

    @property
    def current_theme(self) -> Theme:
        """Get the current theme.

        Returns:
            The current Theme instance.
        """
        return self.theme_manager.current_theme

    @property
    def current_mode(self) -> ThemeMode:
        """Get the current theme mode.

        Returns:
            The current ThemeMode.
        """
        return self.theme_manager.current_theme.mode

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
from gpt_trader.tui.preferences_paths import resolve_preferences_paths
from gpt_trader.tui.theme import Theme, ThemeMode, get_theme_manager
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from textual.app import App

logger = get_logger(__name__, component="tui")


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
        self.preferences_path, self._fallback_path = resolve_preferences_paths(preferences_path)

    def load_preference(self) -> ThemeMode:
        """Load theme preference from config file.

        Returns:
            The loaded theme mode, or DARK as default.
        """
        for path in filter(None, (self.preferences_path, self._fallback_path)):
            try:
                if path.exists():
                    with open(path) as f:
                        prefs = json.load(f)
                        mode_str = prefs.get("theme", "dark")
                        mode = ThemeMode(mode_str)
                        self.theme_manager.set_theme(mode)
                        logger.debug("Loaded theme preference: %s (from %s)", mode_str, path)
                        return mode
            except Exception as e:
                logger.debug("Could not load theme preference from %s: %s", path, e)

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
            source_path = (
                self.preferences_path
                if self.preferences_path.exists()
                else (self._fallback_path if self._fallback_path and self._fallback_path.exists() else None)
            )
            if source_path is not None:
                with open(source_path) as f:
                    prefs = json.load(f)

            prefs["theme"] = mode.value

            with open(self.preferences_path, "w") as f:
                json.dump(prefs, f, indent=2)

            logger.debug("Saved theme preference: %s", mode)
            return True
        except Exception as e:
            logger.warning(f"Could not save theme preference: {e}")
            return False

    def toggle_theme(self) -> Theme:
        """Cycle through dark, light, and high-contrast themes.

        Returns:
            The new theme after cycling.
        """
        current = self.theme_manager.current_mode
        # Cycle: DARK -> LIGHT -> HIGH_CONTRAST -> DARK
        theme_cycle = {
            ThemeMode.DARK: ThemeMode.LIGHT,
            ThemeMode.LIGHT: ThemeMode.HIGH_CONTRAST,
            ThemeMode.HIGH_CONTRAST: ThemeMode.DARK,
        }
        new_mode = theme_cycle.get(current, ThemeMode.DARK)

        apply_ok = True
        try:
            apply_css = getattr(self.app, "apply_theme_css", None)
            if callable(apply_css):
                apply_ok = bool(apply_css(new_mode))
        except Exception as e:
            apply_ok = False
            logger.debug("Theme CSS hot-swap failed: %s", e)

        if not apply_ok:
            self.app.notify(
                "Theme switch failed (missing CSS). Run `python scripts/build_tui_css.py`.",
                severity="warning",
                title="Theme",
            )
            return self.theme_manager.current_theme

        mode_names = {
            ThemeMode.DARK: "Dark",
            ThemeMode.LIGHT: "Light",
            ThemeMode.HIGH_CONTRAST: "High Contrast",
        }
        mode_name = mode_names.get(new_mode, "Dark")
        self.theme_manager.set_theme(new_mode)
        self.save_preference(new_mode)

        try:
            self.app.post_message(ThemeChanged(theme_mode=new_mode.value))
        except Exception:
            pass

        self.app.notify(
            f"Theme set to {mode_name}",
            severity="information",
            title="Theme",
        )

        logger.debug("Theme preference updated to: %s", new_mode.value)
        return self.theme_manager.current_theme

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

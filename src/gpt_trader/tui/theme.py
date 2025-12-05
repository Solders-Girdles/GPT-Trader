"""Theme system for GPT-Trader TUI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ThemeMode(str, Enum):
    """Available theme modes."""

    DARK = "dark"
    LIGHT = "light"


@dataclass(frozen=True)
class ColorPalette:
    """Color palette for TUI themes."""

    # Backgrounds
    background_primary: str
    background_secondary: str
    background_elevated: str
    surface: str

    # Accents
    accent_primary: str
    accent_secondary: str

    # Text
    text_primary: str
    text_secondary: str
    text_muted: str

    # Semantic
    success: str
    warning: str
    error: str
    info: str

    # Interactive states
    overlay_hover: str
    overlay_focus: str
    overlay_disabled: str

    # Borders
    border: str
    border_muted: str


# Dark Theme (Claude Code-inspired warm palette)
DARK_PALETTE = ColorPalette(
    # Backgrounds
    background_primary="#1A1815",  # Deep warm brown
    background_secondary="#2A2520",  # Lighter warm brown
    background_elevated="#3D3833",  # Elevated surfaces
    surface="#2A2520",
    # Accents
    accent_primary="#D4744F",  # Rust-orange
    accent_secondary="#E08A6A",  # Lighter rust
    # Text
    text_primary="#F0EDE9",  # High contrast
    text_secondary="#B8B5B2",  # Medium contrast
    text_muted="#7A7672",  # Low contrast
    # Semantic
    success="#85B77F",  # Warm green
    warning="#E0B366",  # Warm amber
    error="#E08580",  # Warm coral-red
    info="#7FB8D4",  # Cool blue-grey
    # Interactive
    overlay_hover="rgba(193, 95, 60, 0.1)",
    overlay_focus="rgba(193, 95, 60, 0.2)",
    overlay_disabled="rgba(58, 53, 48, 0.5)",
    # Borders
    border="#4A4540",
    border_muted="#3A3530",
)

# Light Theme (Claude Code-inspired, inverted)
LIGHT_PALETTE = ColorPalette(
    # Backgrounds
    background_primary="#FAF8F5",  # Warm off-white
    background_secondary="#F0EDE9",  # Slightly darker
    background_elevated="#FFFFFF",  # Pure white for elevation
    surface="#F0EDE9",
    # Accents
    accent_primary="#C15F3C",  # Darker rust-orange for contrast
    accent_secondary="#D4744F",  # Original rust
    # Text
    text_primary="#1A1815",  # Dark brown (was bg-primary)
    text_secondary="#3D3833",  # Medium brown
    text_muted="#7A7672",  # Muted grey
    # Semantic
    success="#4A7D44",  # Darker green for contrast
    warning="#C89A2E",  # Darker amber
    error="#C14F4A",  # Darker coral-red
    info="#4A7D9A",  # Darker blue
    # Interactive
    overlay_hover="rgba(193, 95, 60, 0.08)",
    overlay_focus="rgba(193, 95, 60, 0.15)",
    overlay_disabled="rgba(122, 118, 114, 0.3)",
    # Borders
    border="#D4CFCA",
    border_muted="#E5E2DE",
)


@dataclass(frozen=True)
class Theme:
    """Complete theme configuration."""

    mode: ThemeMode
    colors: ColorPalette
    spacing_xs: int = 1
    spacing_sm: int = 2
    spacing_md: int = 3
    spacing_lg: int = 4


class ThemeManager:
    """Manages theme selection and switching."""

    def __init__(self, initial_mode: ThemeMode = ThemeMode.DARK):
        self._current_mode = initial_mode
        self._themes = {
            ThemeMode.DARK: Theme(mode=ThemeMode.DARK, colors=DARK_PALETTE),
            ThemeMode.LIGHT: Theme(mode=ThemeMode.LIGHT, colors=LIGHT_PALETTE),
        }

    @property
    def current_theme(self) -> Theme:
        """Get the current active theme."""
        return self._themes[self._current_mode]

    @property
    def current_mode(self) -> ThemeMode:
        """Get the current theme mode."""
        return self._current_mode

    def toggle_theme(self) -> Theme:
        """Toggle between dark and light themes."""
        if self._current_mode == ThemeMode.DARK:
            self._current_mode = ThemeMode.LIGHT
        else:
            self._current_mode = ThemeMode.DARK
        return self.current_theme

    def set_theme(self, mode: ThemeMode) -> Theme:
        """Set a specific theme mode."""
        self._current_mode = mode
        return self.current_theme


# Global theme manager instance
_theme_manager: ThemeManager | None = None


def get_theme_manager() -> ThemeManager:
    """Get or create the global theme manager."""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager


# Backward compatibility - keep THEME for existing code
THEME = Theme(mode=ThemeMode.DARK, colors=DARK_PALETTE)

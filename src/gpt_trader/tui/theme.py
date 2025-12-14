"""Theme system for GPT-Trader TUI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ThemeMode(str, Enum):
    """Available theme modes."""

    DARK = "dark"
    LIGHT = "light"
    HIGH_CONTRAST = "high_contrast"


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


# Dark Theme (Obsidian - FinTech Pro)
DARK_PALETTE = ColorPalette(
    # Backgrounds
    background_primary="#101010",  # Deep Obsidian (Deepest)
    background_secondary="#1a1a1a",  # Obsidian (Lighter)
    background_elevated="#252525",  # Obsidian (Elevated)
    surface="#303030",             # Obsidian (Surface)
    # Accents
    accent_primary="#007AFF",      # Electric Blue - Primary
    accent_secondary="#0056b3",    # Deep Blue - Secondary
    # Text
    text_primary="#FFFFFF",        # Pure White (Brightest)
    text_secondary="#B0B0B0",      # Light Grey (Bright)
    text_muted="#606060",          # Dark Grey (Muted)
    # Semantic
    success="#00FF41",             # Neon Green
    warning="#FFD700",             # Gold
    error="#FF0033",               # Crimson Red
    info="#007AFF",                # Electric Blue
    # Interactive
    overlay_hover="rgba(255, 255, 255, 0.05)",
    overlay_focus="rgba(0, 122, 255, 0.2)",
    overlay_disabled="rgba(0, 0, 0, 0.5)",
    # Borders
    border="#333333",              # Dark Grey
    border_muted="#1a1a1a",        # Obsidian
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

# High Contrast Theme (Accessibility-focused, WCAG AAA compliant)
HIGH_CONTRAST_PALETTE = ColorPalette(
    # Backgrounds - Pure black for maximum contrast
    background_primary="#000000",
    background_secondary="#0a0a0a",
    background_elevated="#1a1a1a",
    surface="#2a2a2a",
    # Accents - Bright cyan for visibility
    accent_primary="#00AAFF",
    accent_secondary="#0088DD",
    # Text - High contrast for readability
    text_primary="#FFFFFF",  # 21:1 contrast on black
    text_secondary="#CCCCCC",  # 13:1 contrast on black
    text_muted="#999999",  # 7:1 contrast on black (WCAG AAA)
    # Semantic - Enhanced saturation
    success="#00FF66",
    warning="#FFCC00",
    error="#FF3366",
    info="#00AAFF",
    # Interactive - Higher opacity for visibility
    overlay_hover="rgba(255, 255, 255, 0.10)",
    overlay_focus="rgba(0, 170, 255, 0.30)",
    overlay_disabled="rgba(0, 0, 0, 0.6)",
    # Borders - More visible
    border="#555555",
    border_muted="#444444",
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
            ThemeMode.HIGH_CONTRAST: Theme(mode=ThemeMode.HIGH_CONTRAST, colors=HIGH_CONTRAST_PALETTE),
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


class ThemeProxy:
    """Proxy that always reflects the current ThemeManager theme.

    This lets existing code keep using `THEME.colors.*` while ensuring
    runtime values stay in sync with the saved theme preference.
    """

    @property
    def _theme(self) -> Theme:
        return get_theme_manager().current_theme

    @property
    def colors(self) -> ColorPalette:
        return self._theme.colors

    @property
    def mode(self) -> ThemeMode:
        return self._theme.mode

    def __getattr__(self, name: str) -> object:
        return getattr(self._theme, name)


# Backward compatibility - keep THEME name for existing imports
THEME = ThemeProxy()

"""Theme constants for GPT-Trader TUI.

This module defines the Claude Code-inspired color palette and theme configuration
for the terminal user interface. All UI components should reference these constants
rather than hardcoding color values.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorPalette:
    """Claude Code-inspired color palette.

    Color Philosophy:
    - Warm, analog aesthetic ("parchment-toned")
    - High contrast without harshness (WCAG AA compliant)
    - Editorial quality with thoughtful spacing
    - Subtle, delightful micro-interactions
    """

    # Backgrounds
    background_primary: str = "#1A1815"  # Deep warm brown "coffee stain"
    background_secondary: str = "#2E2922"  # Slightly lighter warm brown
    background_elevated: str = "#3A3530"  # Elevated surfaces

    # Accents
    accent_primary: str = "#C15F3C"  # Signature rust-orange
    accent_secondary: str = "#D17A5A"  # Lighter rust for hover states
    accent_muted: str = "#8B4428"  # Darker rust for borders

    # Text
    text_primary: str = "#E8E6E3"  # Light warm grey
    text_secondary: str = "#ABA8A5"  # Muted grey for labels
    text_muted: str = "#6B6865"  # Very muted grey for hints

    # Semantic
    success: str = "#7AA874"  # Warm green
    warning: str = "#D8A657"  # Warm amber
    error: str = "#D4736E"  # Warm coral-red
    info: str = "#6FA8C4"  # Warm blue-grey

    # Borders
    border_subtle: str = "#3A3530"  # Subtle warm border
    border_emphasis: str = "#4A453F"  # Emphasized border


@dataclass(frozen=True)
class Theme:
    """Complete theme configuration for the TUI.

    Provides a centralized configuration for colors, typography, spacing,
    and component dimensions throughout the interface.
    """

    colors: ColorPalette

    # Typography
    font_size_sm: int = 12
    font_size_base: int = 14
    font_size_lg: int = 16

    # Spacing (in terminal cells)
    spacing_xs: int = 1
    spacing_sm: int = 2
    spacing_md: int = 3
    spacing_lg: int = 4

    # Component heights
    status_bar_height: int = 6
    footer_height: int = 1
    header_height: int = 3


# Global theme instance
THEME: Theme = Theme(colors=ColorPalette())


def get_color(semantic_name: str) -> str:
    """Get color by semantic name.

    Args:
        semantic_name: Name of the color (e.g., 'success', 'error', 'accent_primary')

    Returns:
        Hex color code

    Raises:
        AttributeError: If semantic_name doesn't exist in ColorPalette

    Example:
        >>> get_color('success')
        '#7AA874'
        >>> get_color('accent_primary')
        '#C15F3C'
    """
    color: str = getattr(THEME.colors, semantic_name)
    return color

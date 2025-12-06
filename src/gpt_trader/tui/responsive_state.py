"""Responsive state enum for TUI layout management.

This module defines the responsive states used throughout the TUI
to adapt the layout based on terminal width.
"""

from enum import Enum


class ResponsiveState(str, Enum):
    """Terminal width-based responsive states.

    The TUI adapts its layout based on terminal width using these states.
    Each state corresponds to a range of terminal column widths:

    Breakpoints:
        - COMPACT: 0-119 cols (minimum viable, hide non-essentials)
        - STANDARD: 120-139 cols (balanced, core metrics visible)
        - COMFORTABLE: 140-159 cols (full experience, all metrics)
        - WIDE: 160+ cols (luxurious, expanded labels)

    Using str as a base class allows enum values to be used directly
    in string comparisons and CSS class generation.
    """

    COMPACT = "compact"
    STANDARD = "standard"
    COMFORTABLE = "comfortable"
    WIDE = "wide"

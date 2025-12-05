"""Responsive design utilities for TUI.

This module provides utilities for handling different terminal widths,
including breakpoint calculation and dynamic width sizing.
"""


def calculate_modal_width(terminal_width: int, size: str = "medium") -> int:
    """Calculate modal width based on terminal size.

    Ensures modals have appropriate width with margins, scaling
    from minimum to maximum based on terminal width percentage.

    Args:
        terminal_width: Current terminal width in columns
        size: Modal size category:
            - "small": 40-60 cols (60% of terminal)
            - "medium": 60-80 cols (70% of terminal)
            - "large": 80-120 cols (80% of terminal)

    Returns:
        Width in columns with 20-column margins enforced

    Examples:
        >>> calculate_modal_width(100, "small")
        60
        >>> calculate_modal_width(80, "small")
        48
        >>> calculate_modal_width(200, "small")
        60  # Capped at max
    """
    config = {
        "small": {"max": 60, "min": 40, "pct": 0.6},
        "medium": {"max": 80, "min": 60, "pct": 0.7},
        "large": {"max": 120, "min": 80, "pct": 0.8},
    }

    c = config[size]
    calculated = int(terminal_width * c["pct"])

    # Ensure 20-column margins (10 on each side)
    max_allowed = terminal_width - 20

    return max(c["min"], min(c["max"], calculated, max_allowed))


def calculate_responsive_state(width: int) -> str:
    """Calculate responsive state based on terminal width.

    Args:
        width: Terminal width in columns

    Returns:
        Responsive state: "compact", "standard", "comfortable", or "wide"

    Breakpoints:
        - compact: 0-119 cols (minimum viable, hide non-essentials)
        - standard: 120-139 cols (balanced, core metrics visible)
        - comfortable: 140-159 cols (full experience, all metrics)
        - wide: 160+ cols (luxurious, expanded labels)
    """
    if width < 120:
        return "compact"
    elif width < 140:
        return "standard"
    elif width < 160:
        return "comfortable"
    else:
        return "wide"

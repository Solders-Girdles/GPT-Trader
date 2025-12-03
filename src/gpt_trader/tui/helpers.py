"""TUI Utilities."""

import atexit
import functools
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from textual.app import App

logger = get_logger(__name__, component="tui")

F = TypeVar("F", bound=Callable[..., Any])

# Escape sequences that disable the mouse tracking / bracketed paste modes
# Textual enables. When the app exits unexpectedly, these can leak into the
# shell and cause raw mouse coordinates to print while you move the cursor.
_TERMINAL_RESET_SEQUENCE = (
    "\x1b[?1000l"  # Disable mouse click tracking
    "\x1b[?1002l"  # Disable mouse drag tracking
    "\x1b[?1003l"  # Disable all-motion tracking
    "\x1b[?1004l"  # Disable focus in/out events
    "\x1b[?1005l"  # Disable UTF-8 mouse mode
    "\x1b[?1006l"  # Disable SGR mouse mode
    "\x1b[?1015l"  # Disable urxvt mouse mode
    "\x1b[?2004l"  # Disable bracketed paste
    "\x1b[?25h"  # Ensure cursor is visible again
)

_terminal_cleanup_registered = False


def reset_terminal_modes() -> None:
    """Best-effort reset of terminal modes that can leak after TUI exit."""
    try:
        sys.stdout.write(_TERMINAL_RESET_SEQUENCE)
        sys.stdout.flush()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to reset terminal state: %s", exc)


def run_tui_app_with_cleanup(app: "App") -> None:
    """
    Run a Textual app and always reset terminal state on exit.

    Guards against mouse-tracking escape codes appearing in the shell if the
    TUI exits abruptly.
    """

    global _terminal_cleanup_registered
    if not _terminal_cleanup_registered:
        atexit.register(reset_terminal_modes)
        _terminal_cleanup_registered = True

    try:
        app.run()
    finally:
        reset_terminal_modes()


def safe_update(func: F) -> F:
    """
    Decorator to wrap widget update methods.
    Catches exceptions, logs them, and prevents the app from crashing.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Try to get class name if it's a method
            context = func.__name__
            if args and hasattr(args[0], "__class__"):
                context = f"{args[0].__class__.__name__}.{func.__name__}"

            logger.error(f"Error in TUI update ({context}): {e}", exc_info=True)
            return None

    return wrapper  # type: ignore

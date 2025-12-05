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


def safe_update(
    func: F | None = None,
    *,
    notify_user: bool = False,
    severity: str = "warning",
    error_tracker: bool = False,
) -> F:
    """
    Decorator to wrap widget update methods with error handling.

    Catches exceptions, logs them, and optionally notifies users.
    Prevents TUI crashes from individual widget update failures.

    Args:
        func: Function to wrap (provided automatically when used as @safe_update)
        notify_user: If True, show toast notification to user on error
        severity: Notification severity ("information", "warning", "error")
        error_tracker: If True, register error with ErrorIndicatorWidget

    Returns:
        Wrapped function that safely handles exceptions

    Examples:
        Basic usage (backward compatible)::

            @safe_update
            def update_positions(self, ...):
                ...

        With user notification::

            @safe_update(notify_user=True, severity="warning")
            def update_market_data(self, ...):
                ...

        With error tracking::

            @safe_update(notify_user=True, error_tracker=True)
            def update_account(self, ...):
                ...
    """

    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                # Get context for logging
                context = f.__name__
                widget_class = None
                if args and hasattr(args[0], "__class__"):
                    widget_class = args[0].__class__.__name__
                    context = f"{widget_class}.{f.__name__}"

                # Always log with full traceback
                logger.error(f"Error in TUI update ({context}): {e}", exc_info=True)

                # Optionally notify user via toast
                if notify_user and args and hasattr(args[0], "app"):
                    try:
                        app = args[0].app
                        if app and hasattr(app, "notify"):
                            app.notify(
                                f"Update failed: {context}",
                                title="Widget Update Error",
                                severity=severity,
                                timeout=5,
                            )
                    except Exception:
                        # Don't let notification failure crash the error handler
                        pass

                # Optionally register with error tracker widget
                if error_tracker and args and hasattr(args[0], "app"):
                    try:
                        app = args[0].app
                        if app and hasattr(app, "error_tracker"):
                            app.error_tracker.add_error(
                                widget=widget_class or "Unknown",
                                method=f.__name__,
                                error=str(e),
                            )
                    except Exception:
                        # Don't let tracker registration crash the error handler
                        pass

                return None

        return wrapper  # type: ignore

    # Support both @safe_update and @safe_update(notify_user=True)
    if func is None:
        # Called with parentheses: @safe_update(...)
        return decorator  # type: ignore[return-value]
    else:
        # Called without parentheses: @safe_update
        return decorator(func)

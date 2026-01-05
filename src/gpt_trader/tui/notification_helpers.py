"""Shared notification helpers for consistent TUI messaging.

Provides unified severity mapping, standard notification formats, and
recovery action hints for a coherent alert/error user experience.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from textual.app import App

# Notification severity levels aligned with Textual and AlertManager
NotificationSeverity = Literal["information", "warning", "error"]

# Standard timeout values
TIMEOUT_BRIEF = 2  # Quick acknowledgment (e.g., "Copied")
TIMEOUT_NORMAL = 5  # Standard notifications
TIMEOUT_ALERT = 10  # Important alerts that need attention


def notify_success(
    app: App,
    message: str,
    *,
    title: str | None = None,
    timeout: int = TIMEOUT_BRIEF,
) -> None:
    """Show a success notification.

    Args:
        app: The Textual app instance.
        message: Success message to display.
        title: Optional title for the notification.
        timeout: Notification timeout in seconds.
    """
    app.notify(message, title=title, severity="information", timeout=timeout)


def notify_action(
    app: App,
    message: str,
    *,
    title: str | None = None,
    timeout: int = TIMEOUT_NORMAL,
) -> None:
    """Show an action/status notification.

    Args:
        app: The Textual app instance.
        message: Action message to display.
        title: Optional title for the notification.
        timeout: Notification timeout in seconds.
    """
    app.notify(message, title=title, severity="information", timeout=timeout)


def notify_warning(
    app: App,
    message: str,
    *,
    title: str | None = None,
    recovery_hint: str | None = None,
    timeout: int = TIMEOUT_NORMAL,
) -> None:
    """Show a warning notification.

    Args:
        app: The Textual app instance.
        message: Warning message to display.
        title: Optional title for the notification.
        recovery_hint: Optional recovery action hint to append.
        timeout: Notification timeout in seconds.
    """
    full_message = f"{message} — {recovery_hint}" if recovery_hint else message
    app.notify(full_message, title=title, severity="warning", timeout=timeout)


def notify_error(
    app: App,
    message: str,
    *,
    title: str | None = None,
    recovery_hint: str | None = None,
    timeout: int = TIMEOUT_ALERT,
) -> None:
    """Show an error notification.

    Args:
        app: The Textual app instance.
        message: Error message to display.
        title: Optional title for the notification.
        recovery_hint: Optional recovery action hint to append.
        timeout: Notification timeout in seconds.
    """
    full_message = f"{message} — {recovery_hint}" if recovery_hint else message
    app.notify(full_message, title=title, severity="error", timeout=timeout)


# Standard recovery hints for common scenarios
RECOVERY_HINTS = {
    # Connection issues
    "connection_lost": "Press R to reconnect",
    "connection_error": "Press R to reconnect",
    "reconnecting": "Reconnecting...",

    # Rate limiting
    "rate_limit": "Reduce request frequency",
    "throttled": "Wait and retry",

    # Risk management
    "reduce_only": "Check risk settings",
    "daily_loss": "Consider pausing trading",
    "position_limit": "Close positions to continue",

    # Configuration
    "config_error": "Press C to check config",
    "invalid_config": "Press C to fix config",

    # Bot state
    "bot_stopped": "Press S to start",
    "bot_error": "Check logs for details",
}


def get_recovery_hint(scenario: str) -> str | None:
    """Get a standard recovery hint for a scenario.

    Args:
        scenario: The scenario identifier (e.g., "connection_lost").

    Returns:
        Recovery hint string or None if no hint available.
    """
    return RECOVERY_HINTS.get(scenario)


def format_error_notification(
    error_type: str,
    message: str,
    *,
    include_hint: bool = True,
) -> tuple[str, str | None]:
    """Format an error notification with optional recovery hint.

    Args:
        error_type: Type of error for hint lookup.
        message: The error message.
        include_hint: Whether to include recovery hint.

    Returns:
        Tuple of (formatted_message, recovery_hint or None).
    """
    hint = get_recovery_hint(error_type) if include_hint else None
    return message, hint

"""
Event Handler Mixin for TUI Widgets.

Provides common event handling patterns for widgets that need to respond
to TUI events. This mixin can be used with any Textual widget to add
standardized event handling behavior.

Usage:
    class MyWidget(EventHandlerMixin, Static):
        def on_mount(self) -> None:
            # Mixin provides default handlers
            pass

        def on_bot_state_changed(self, event: BotStateChanged) -> None:
            # Override to customize behavior
            super().on_bot_state_changed(event)
            # Custom logic here
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.events import (
        BotModeChanged,
        BotStateChanged,
        ErrorOccurred,
        ResponsiveStateChanged,
        StateDeltaUpdateApplied,
        StateUpdateReceived,
        StateValidationFailed,
        StateValidationPassed,
        ThemeChanged,
    )

logger = get_logger(__name__, component="tui")


class EventHandlerMixin:
    """
    Mixin providing default event handler implementations.

    This mixin provides no-op default implementations for common TUI events.
    Widgets can inherit from this mixin and override only the events they
    care about, avoiding boilerplate for unused events.

    All handlers log at debug level by default for troubleshooting.
    """

    # ==========================================================================
    # Bot Lifecycle Event Handlers
    # ==========================================================================

    def on_bot_state_changed(self, event: BotStateChanged) -> None:
        """
        Handle bot state change (started/stopped).

        Default implementation logs the state change. Override to customize.

        Args:
            event: Contains running state and uptime
        """
        logger.debug(
            f"{self.__class__.__name__} received BotStateChanged: "
            f"running={event.running}, uptime={event.uptime:.1f}s"
        )

    def on_bot_mode_changed(self, event: BotModeChanged) -> None:
        """
        Handle bot mode change (demo, paper, read_only, live).

        Default implementation logs the mode change. Override to customize.

        Args:
            event: Contains new_mode and old_mode
        """
        logger.debug(
            f"{self.__class__.__name__} received BotModeChanged: "
            f"{event.old_mode} → {event.new_mode}"
        )

    # ==========================================================================
    # State Update Event Handlers
    # ==========================================================================

    def on_state_update_received(self, event: StateUpdateReceived) -> None:
        """
        Handle new state update from StatusReporter.

        Default implementation logs receipt. Override to process updates.

        Args:
            event: Contains BotStatus snapshot
        """
        logger.debug(
            f"{self.__class__.__name__} received StateUpdateReceived "
            f"(status timestamp: {event.status.market.last_price_update if hasattr(event.status, 'market') else 'unknown'})"
        )

    def on_state_validation_failed(self, event: StateValidationFailed) -> None:
        """
        Handle state validation failure.

        Default implementation logs validation errors. Override to display
        errors to user or take corrective action.

        Args:
            event: Contains list of validation errors
        """
        logger.warning(
            f"{self.__class__.__name__} received StateValidationFailed: "
            f"{len(event.errors)} error(s) in component '{event.component}'"
        )
        for error in event.errors:
            logger.warning(f"  - {error.field}: {error.message} " f"(severity: {error.severity})")

    def on_state_validation_passed(self, event: StateValidationPassed) -> None:
        """
        Handle successful state validation.

        Default implementation logs success. Override to clear error displays.

        Args:
            event: State validation passed event
        """
        logger.debug(f"{self.__class__.__name__} received StateValidationPassed")

    def on_state_delta_update_applied(self, event: StateDeltaUpdateApplied) -> None:
        """
        Handle delta update application.

        Default implementation logs components updated. Override for debugging.

        Args:
            event: Contains list of updated components
        """
        logger.debug(
            f"{self.__class__.__name__} received StateDeltaUpdateApplied: "
            f"components={', '.join(event.components_updated)}, "
            f"full_update={event.use_full_update}"
        )

    # ==========================================================================
    # UI Coordination Event Handlers
    # ==========================================================================

    def on_responsive_state_changed(self, event: ResponsiveStateChanged) -> None:
        """
        Handle terminal resize and responsive state change.

        Default implementation logs state change. Override to adjust layout.

        Args:
            event: Contains new state and terminal width
        """
        logger.debug(
            f"{self.__class__.__name__} received ResponsiveStateChanged: "
            f"state={event.state}, width={event.width}"
        )

    def on_theme_changed(self, event: ThemeChanged) -> None:
        """
        Handle theme change (light/dark).

        Default implementation logs theme change. Override to update colors.

        Args:
            event: Contains new theme mode
        """
        logger.debug(
            f"{self.__class__.__name__} received ThemeChanged: " f"mode={event.theme_mode}"
        )

    # ==========================================================================
    # Error Event Handlers
    # ==========================================================================

    def on_error_occurred(self, event: ErrorOccurred) -> None:
        """
        Handle error event.

        Default implementation logs the error. Override to display to user.

        Args:
            event: Contains error message, severity, and context
        """
        log_level = "error" if event.severity == "error" else "warning"
        log_method = getattr(logger, log_level)
        log_method(
            f"{self.__class__.__name__} received ErrorOccurred: "
            f"[{event.severity}] {event.message} "
            f"(context: {event.context})"
        )
        if event.exception:
            logger.debug(f"Exception: {event.exception}", exc_info=event.exception)

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def post_event(self, event: object) -> None:
        """
        Convenience method to post an event.

        Safely posts event if widget has post_message method (i.e., is mounted).

        Args:
            event: Event to post
        """
        if hasattr(self, "post_message"):
            self.post_message(event)  # type: ignore[attr-defined]
        else:
            logger.warning(
                f"{self.__class__.__name__} tried to post event but "
                f"post_message not available (widget not mounted?)"
            )

    def log_event_received(self, event_name: str, details: str = "") -> None:
        """
        Convenience method to log event receipt.

        Args:
            event_name: Name of the event
            details: Additional details to log
        """
        message = f"{self.__class__.__name__} received {event_name}"
        if details:
            message += f": {details}"
        logger.debug(message)

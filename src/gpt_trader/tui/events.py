"""
TUI Event System.

This module defines all custom events used in the TUI for decoupled communication
between components. Events replace direct widget queries and enable loose coupling
between managers, screens, and widgets.

Events are organized by domain:
- Bot Lifecycle: Start, stop, state changes
- State Updates: Data updates and validation
- UI Coordination: Refresh requests, heartbeat
- Trade Matching: Reset and state requests

Usage:
    # In manager: Post an event
    self.app.post_message(BotStartRequested())

    # In widget: Handle the event
    def on_bot_start_requested(self, event: BotStartRequested) -> None:
        # Handle the event
        pass
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from textual.message import Message

from gpt_trader.tui.responsive_state import ResponsiveState

if TYPE_CHECKING:
    from gpt_trader.monitoring.status_reporter import BotStatus


# ==============================================================================
# Bot Lifecycle Events
# ==============================================================================


class BotStartRequested(Message):
    """
    Request to start the trading bot.

    Posted by: UI controls, action handlers
    Handled by: BotLifecycleManager

    This event is a request that may be denied (e.g., if bot is already running).
    Listeners should watch for BotStateChanged to confirm actual state change.
    """


class BotStopRequested(Message):
    """
    Request to stop the trading bot.

    Posted by: UI controls, action handlers, panic handler
    Handled by: BotLifecycleManager

    This event is a request that may be denied (e.g., if bot is already stopped).
    Listeners should watch for BotStateChanged to confirm actual state change.
    """


@dataclass
class BotStateChanged(Message):
    """
    Bot running state has changed.

    Posted by: BotLifecycleManager after start/stop operations
    Handled by: Widgets that display bot state (BotStatusWidget, etc.)

    Attributes:
        running: True if bot is now running, False if stopped
        uptime: Current uptime in seconds (0 if stopped)
    """

    running: bool
    uptime: float = 0.0


@dataclass
class BotModeChangeRequested(Message):
    """
    Request to switch bot mode (demo, paper, read_only, live).

    Posted by: ModeSelector widget
    Handled by: BotLifecycleManager

    Attributes:
        target_mode: Mode to switch to ("demo", "paper", "read_only", "live")
    """

    target_mode: str


@dataclass
class BotModeChanged(Message):
    """
    Bot mode has been changed successfully.

    Posted by: BotLifecycleManager after mode switch completes
    Handled by: Widgets that display mode (ModeIndicator, etc.)

    Attributes:
        new_mode: The new mode ("demo", "paper", "read_only", "live")
        old_mode: The previous mode
    """

    new_mode: str
    old_mode: str


# ==============================================================================
# State Update Events
# ==============================================================================


@dataclass
class StateUpdateReceived(Message):
    """
    New state update received from bot's StatusReporter.

    Posted by: UICoordinator when observer callback fires
    Handled by: TuiState for validation and propagation

    Attributes:
        status: Typed BotStatus snapshot from StatusReporter
        runtime_state: Optional runtime state (uptime, cycle count, etc.)
    """

    status: BotStatus
    runtime_state: Any | None = None


@dataclass
class ValidationError(Message):
    """
    Details of a single state validation error.

    Used as a component of StateValidationFailed event.

    Attributes:
        field: Field name that failed validation
        message: Human-readable error message
        severity: "warning" or "error"
        value: The invalid value (for debugging)
    """

    field: str
    message: str
    severity: str = "error"  # "warning" or "error"
    value: Any = None


@dataclass
class StateValidationFailed(Message):
    """
    State validation found errors.

    Posted by: TuiState after validation layer detects issues
    Handled by: ValidationIndicatorWidget, ErrorIndicatorWidget

    Attributes:
        errors: List of validation errors found
        component: Component that failed validation (e.g., "market", "positions")
    """

    errors: list[ValidationError]
    component: str = "unknown"


class StateValidationPassed(Message):
    """
    State validation completed successfully.

    Posted by: TuiState after successful validation
    Handled by: ValidationIndicatorWidget (to clear warnings)

    This event indicates that the latest state update passed all validation
    checks and can be safely propagated to UI widgets.
    """


@dataclass
class StateDeltaUpdateApplied(Message):
    """
    State delta update has been applied (not full replacement).

    Posted by: TuiState after applying delta updates
    Handled by: Debugging/logging components

    Attributes:
        components_updated: List of component names that were updated
        use_full_update: True if fell back to full update due to errors
    """

    components_updated: list[str]
    use_full_update: bool = False


# ==============================================================================
# UI Coordination Events
# ==============================================================================


class UIRefreshRequested(Message):
    """
    Request immediate UI refresh.

    Posted by: Action handlers, reconnect operations
    Handled by: UICoordinator, MainScreen

    This triggers an out-of-band UI update, bypassing the normal update loop.
    Useful for user-initiated reconnections or manual refresh.
    """


@dataclass
class HeartbeatTick(Message):
    """
    Periodic heartbeat tick for animations.

    Posted by: UICoordinator heartbeat loop (every 1 second)
    Handled by: BotStatusWidget for pulse animation

    Attributes:
        pulse_value: Sine wave value 0.0-1.0 for smooth animation
    """

    pulse_value: float = 0.0


@dataclass
class ResponsiveStateChanged(Message):
    """
    Terminal width changed, responsive state updated.

    Posted by: ResponsiveManager on resize events
    Handled by: MainScreen, widgets with responsive layouts

    Attributes:
        state: New responsive state (ResponsiveState enum)
        width: New terminal width in columns
    """

    state: ResponsiveState
    width: int


# ==============================================================================
# Widget Control Events
# ==============================================================================


@dataclass
class ModeSelectorEnabledChanged(Message):
    """
    Request to enable or disable the mode selector widget.

    Posted by: BotLifecycleManager on bot start/stop
    Handled by: ModeSelector widget

    Attributes:
        enabled: True to enable, False to disable
    """

    enabled: bool


@dataclass
class ModeSelectorValueChanged(Message):
    """
    Request to update the mode selector's current value.

    Posted by: BotLifecycleManager after mode switch
    Handled by: ModeSelector widget

    Attributes:
        mode: New mode value ("demo", "paper", "read_only", "live")
    """

    mode: str


@dataclass
class ModeSelectorLoadingChanged(Message):
    """
    Request to update the mode selector's loading state.

    Posted by: BotLifecycleManager during mode switch operations
    Handled by: ModeSelector widget

    Attributes:
        loading: True to show loading spinner, False to hide
    """

    loading: bool


class MainScreenRefreshRequested(Message):
    """
    Request the main screen to refresh its UI state.

    Posted by: BotLifecycleManager after bot lifecycle operations
    Handled by: MainScreen

    This event replaces direct MainScreen.update_ui() calls from managers.
    """


# ==============================================================================
# Trade Matching Events
# ==============================================================================


class TradeMatcherResetRequested(Message):
    """
    Request to reset trade matcher state.

    Posted by: BotLifecycleManager on bot start or mode switch
    Handled by: TradesWidget

    This event replaces direct access to TradesWidget._trade_matcher.reset().
    Ensures clean P&L tracking when starting fresh or switching modes.
    """


@dataclass
class TradeMatcherStateRequest(Message):
    """
    Request current trade matcher state.

    Posted by: Debug screens, system details
    Handled by: TradesWidget

    Attributes:
        request_id: Unique ID to match response to request
    """

    request_id: str


@dataclass
class TradeMatcherStateResponse(Message):
    """
    Response with current trade matcher state.

    Posted by: TradesWidget in response to TradeMatcherStateRequest
    Handled by: Component that requested state

    Attributes:
        request_id: ID from the request
        state: Dictionary containing trade matcher state
    """

    request_id: str
    state: dict[str, Any]


# ==============================================================================
# Error and Notification Events
# ==============================================================================


@dataclass
class ErrorOccurred(Message):
    """
    An error occurred that should be displayed to the user.

    Posted by: Any component encountering errors
    Handled by: ErrorIndicatorWidget

    Attributes:
        message: Error message to display
        severity: "warning" or "error"
        context: Additional context (component name, operation, etc.)
        exception: Optional exception object for debugging
    """

    message: str
    severity: str = "error"
    context: str = ""
    exception: Exception | None = None


@dataclass
class NotificationRequested(Message):
    """
    Request to show a notification to the user.

    Posted by: Services, managers, widgets
    Handled by: TraderApp (uses app.notify())

    Attributes:
        message: Notification message
        title: Optional title
        severity: "information", "warning", or "error"
        timeout: Timeout in seconds (None for no timeout)
    """

    message: str
    title: str = ""
    severity: str = "information"
    timeout: int | None = None


# ==============================================================================
# Configuration Events
# ==============================================================================


class ConfigReloadRequested(Message):
    """
    Request to reload bot configuration.

    Posted by: ConfigModal, config action handlers
    Handled by: ConfigService
    """


@dataclass
class ConfigChanged(Message):
    """
    Bot configuration has been changed.

    Posted by: ConfigService after successful config update
    Handled by: Widgets that display config (ConfigModal refresh, etc.)

    Attributes:
        config: The new configuration object
    """

    config: Any  # BotConfig type, but avoid circular import


# ==============================================================================
# Theme Events
# ==============================================================================


@dataclass
class ThemeChangeRequested(Message):
    """
    Request to change theme.

    Posted by: Theme toggle action, settings
    Handled by: ThemeService

    Attributes:
        theme_mode: "light" or "dark"
    """

    theme_mode: str


@dataclass
class ThemeChanged(Message):
    """
    Theme has been changed.

    Posted by: ThemeService after theme switch
    Handled by: Widgets that need to update for new theme

    Attributes:
        theme_mode: New theme mode ("light" or "dark")
    """

    theme_mode: str

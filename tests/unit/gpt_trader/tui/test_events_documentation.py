"""Tests that TUI event classes are documented."""

from __future__ import annotations

from gpt_trader.tui.events import (
    BotModeChanged,
    BotModeChangeRequested,
    BotStartRequested,
    BotStateChanged,
    BotStopRequested,
    ConfigChanged,
    ConfigReloadRequested,
    ErrorOccurred,
    FieldValidationError,
    HeartbeatTick,
    NotificationRequested,
    ResponsiveStateChanged,
    StateDeltaUpdateApplied,
    StateUpdateReceived,
    StateValidationFailed,
    StateValidationPassed,
    ThemeChanged,
    ThemeChangeRequested,
    TradeMatcherResetRequested,
    TradeMatcherStateRequest,
    TradeMatcherStateResponse,
    UIRefreshRequested,
)


class TestEventDocumentation:
    """Test that events have proper documentation."""

    def test_all_events_have_docstrings(self):
        """Verify all event classes have docstrings."""
        event_classes = [
            BotStartRequested,
            BotStopRequested,
            BotStateChanged,
            BotModeChangeRequested,
            BotModeChanged,
            StateUpdateReceived,
            FieldValidationError,
            StateValidationFailed,
            StateValidationPassed,
            StateDeltaUpdateApplied,
            UIRefreshRequested,
            HeartbeatTick,
            ResponsiveStateChanged,
            TradeMatcherResetRequested,
            TradeMatcherStateRequest,
            TradeMatcherStateResponse,
            ErrorOccurred,
            NotificationRequested,
            ConfigReloadRequested,
            ConfigChanged,
            ThemeChangeRequested,
            ThemeChanged,
        ]

        for event_cls in event_classes:
            assert event_cls.__doc__ is not None, f"{event_cls.__name__} missing docstring"
            assert len(event_cls.__doc__.strip()) > 0, f"{event_cls.__name__} has empty docstring"

    def test_dataclass_events_have_field_docs(self):
        """Verify dataclass events document their fields."""
        assert "running" in BotStateChanged.__doc__
        assert "uptime" in BotStateChanged.__doc__

        assert "target_mode" in BotModeChangeRequested.__doc__

        assert "errors" in StateValidationFailed.__doc__
        assert "component" in StateValidationFailed.__doc__

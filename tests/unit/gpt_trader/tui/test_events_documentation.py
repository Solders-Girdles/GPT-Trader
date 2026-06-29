"""Tests that TUI event classes are documented."""

from __future__ import annotations

from gpt_trader.tui.events import (
    BotModeChanged,
    BotStartRequested,
    BotStateChanged,
    ConfigChanged,
    ConfigReloadRequested,
    ErrorOccurred,
    FieldValidationError,
    ResponsiveStateChanged,
    StateDeltaUpdateApplied,
    StateUpdateReceived,
    StateValidationFailed,
    StateValidationPassed,
    ThemeChanged,
    TradeMatcherResetRequested,
    TradeMatcherStateRequest,
)


class TestEventDocumentation:
    """Test that events have proper documentation."""

    def test_all_events_have_docstrings(self):
        """Verify all event classes have docstrings."""
        event_classes = [
            BotStartRequested,
            BotStateChanged,
            BotModeChanged,
            StateUpdateReceived,
            FieldValidationError,
            StateValidationFailed,
            StateValidationPassed,
            StateDeltaUpdateApplied,
            ResponsiveStateChanged,
            TradeMatcherResetRequested,
            TradeMatcherStateRequest,
            ErrorOccurred,
            ConfigReloadRequested,
            ConfigChanged,
            ThemeChanged,
        ]

        for event_cls in event_classes:
            assert event_cls.__doc__ is not None, f"{event_cls.__name__} missing docstring"
            assert len(event_cls.__doc__.strip()) > 0, f"{event_cls.__name__} has empty docstring"

    def test_dataclass_events_have_field_docs(self):
        """Verify dataclass events document their fields."""
        assert "running" in BotStateChanged.__doc__
        assert "uptime" in BotStateChanged.__doc__

        assert "errors" in StateValidationFailed.__doc__
        assert "component" in StateValidationFailed.__doc__

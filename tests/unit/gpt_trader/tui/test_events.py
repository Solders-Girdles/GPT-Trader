"""
Tests for TUI event system.

Tests verify that all events:
1. Can be created with required and optional parameters
2. Have correct attribute values
3. Are properly subclassed from Message
4. Can be posted and received in Textual app
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import Static

from gpt_trader.tui.events import (
    BotModeChanged,
    BotModeChangeRequested,
    BotStartRequested,
    BotStateChanged,
    BotStopRequested,
    ConfigChanged,
    ConfigReloadRequested,
    ErrorOccurred,
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
    ValidationError,
)
from gpt_trader.tui.responsive_state import ResponsiveState


class TestBotLifecycleEvents:
    """Test bot lifecycle events."""

    def test_bot_start_requested_creation(self):
        """Test BotStartRequested event creation."""
        event = BotStartRequested()
        assert isinstance(event, Message)

    def test_bot_stop_requested_creation(self):
        """Test BotStopRequested event creation."""
        event = BotStopRequested()
        assert isinstance(event, Message)

    def test_bot_state_changed_creation(self):
        """Test BotStateChanged event creation with attributes."""
        event = BotStateChanged(running=True, uptime=123.45)
        assert isinstance(event, Message)
        assert event.running is True
        assert event.uptime == 123.45

    def test_bot_state_changed_default_uptime(self):
        """Test BotStateChanged with default uptime."""
        event = BotStateChanged(running=False)
        assert event.running is False
        assert event.uptime == 0.0

    def test_bot_mode_change_requested_creation(self):
        """Test BotModeChangeRequested event creation."""
        event = BotModeChangeRequested(target_mode="paper")
        assert isinstance(event, Message)
        assert event.target_mode == "paper"

    def test_bot_mode_changed_creation(self):
        """Test BotModeChanged event creation."""
        event = BotModeChanged(new_mode="live", old_mode="demo")
        assert isinstance(event, Message)
        assert event.new_mode == "live"
        assert event.old_mode == "demo"


class TestStateUpdateEvents:
    """Test state update and validation events."""

    def test_state_update_received_creation(self):
        """Test StateUpdateReceived event creation."""
        mock_status = MagicMock()
        mock_runtime = MagicMock()
        event = StateUpdateReceived(status=mock_status, runtime_state=mock_runtime)
        assert isinstance(event, Message)
        assert event.status == mock_status
        assert event.runtime_state == mock_runtime

    def test_state_update_received_without_runtime(self):
        """Test StateUpdateReceived without runtime state."""
        mock_status = MagicMock()
        event = StateUpdateReceived(status=mock_status)
        assert event.status == mock_status
        assert event.runtime_state is None

    def test_validation_error_creation(self):
        """Test ValidationError event creation."""
        event = ValidationError(
            field="market_data", message="Price cannot be negative", severity="error", value=-10.0
        )
        assert isinstance(event, Message)
        assert event.field == "market_data"
        assert event.message == "Price cannot be negative"
        assert event.severity == "error"
        assert event.value == -10.0

    def test_validation_error_default_severity(self):
        """Test ValidationError with default severity."""
        event = ValidationError(field="test", message="Test error")
        assert event.severity == "error"
        assert event.value is None

    def test_state_validation_failed_creation(self):
        """Test StateValidationFailed event creation."""
        errors = [
            ValidationError(field="field1", message="Error 1"),
            ValidationError(field="field2", message="Error 2"),
        ]
        event = StateValidationFailed(errors=errors, component="positions")
        assert isinstance(event, Message)
        assert len(event.errors) == 2
        assert event.component == "positions"

    def test_state_validation_failed_default_component(self):
        """Test StateValidationFailed with default component."""
        event = StateValidationFailed(errors=[])
        assert event.component == "unknown"

    def test_state_validation_passed_creation(self):
        """Test StateValidationPassed event creation."""
        event = StateValidationPassed()
        assert isinstance(event, Message)

    def test_state_delta_update_applied_creation(self):
        """Test StateDeltaUpdateApplied event creation."""
        components = ["market", "positions", "orders"]
        event = StateDeltaUpdateApplied(components_updated=components, use_full_update=False)
        assert isinstance(event, Message)
        assert event.components_updated == components
        assert event.use_full_update is False

    def test_state_delta_update_applied_full_update(self):
        """Test StateDeltaUpdateApplied with full update fallback."""
        event = StateDeltaUpdateApplied(components_updated=[], use_full_update=True)
        assert event.use_full_update is True


class TestUICoordinationEvents:
    """Test UI coordination events."""

    def test_ui_refresh_requested_creation(self):
        """Test UIRefreshRequested event creation."""
        event = UIRefreshRequested()
        assert isinstance(event, Message)

    def test_heartbeat_tick_creation(self):
        """Test HeartbeatTick event creation."""
        event = HeartbeatTick(pulse_value=0.75)
        assert isinstance(event, Message)
        assert event.pulse_value == 0.75

    def test_heartbeat_tick_default_pulse(self):
        """Test HeartbeatTick with default pulse value."""
        event = HeartbeatTick()
        assert event.pulse_value == 0.0

    def test_responsive_state_changed_creation(self):
        """Test ResponsiveStateChanged event creation."""
        event = ResponsiveStateChanged(state=ResponsiveState.COMFORTABLE, width=140)
        assert isinstance(event, Message)
        assert event.state == ResponsiveState.COMFORTABLE
        assert event.width == 140


class TestTradeMatchingEvents:
    """Test trade matching events."""

    def test_trade_matcher_reset_requested_creation(self):
        """Test TradeMatcherResetRequested event creation."""
        event = TradeMatcherResetRequested()
        assert isinstance(event, Message)

    def test_trade_matcher_state_request_creation(self):
        """Test TradeMatcherStateRequest event creation."""
        event = TradeMatcherStateRequest(request_id="test-123")
        assert isinstance(event, Message)
        assert event.request_id == "test-123"

    def test_trade_matcher_state_response_creation(self):
        """Test TradeMatcherStateResponse event creation."""
        state_data = {"total_pnl": 150.0, "trade_count": 5}
        event = TradeMatcherStateResponse(request_id="test-123", state=state_data)
        assert isinstance(event, Message)
        assert event.request_id == "test-123"
        assert event.state == state_data


class TestErrorNotificationEvents:
    """Test error and notification events."""

    def test_error_occurred_creation(self):
        """Test ErrorOccurred event creation."""
        exc = ValueError("Test error")
        event = ErrorOccurred(
            message="An error occurred", severity="error", context="test_module", exception=exc
        )
        assert isinstance(event, Message)
        assert event.message == "An error occurred"
        assert event.severity == "error"
        assert event.context == "test_module"
        assert event.exception == exc

    def test_error_occurred_defaults(self):
        """Test ErrorOccurred with default values."""
        event = ErrorOccurred(message="Test error")
        assert event.severity == "error"
        assert event.context == ""
        assert event.exception is None

    def test_notification_requested_creation(self):
        """Test NotificationRequested event creation."""
        event = NotificationRequested(
            message="Test notification", title="Test", severity="information", timeout=5
        )
        assert isinstance(event, Message)
        assert event.message == "Test notification"
        assert event.title == "Test"
        assert event.severity == "information"
        assert event.timeout == 5

    def test_notification_requested_defaults(self):
        """Test NotificationRequested with defaults."""
        event = NotificationRequested(message="Test")
        assert event.title == ""
        assert event.severity == "information"
        assert event.timeout is None


class TestConfigurationEvents:
    """Test configuration events."""

    def test_config_reload_requested_creation(self):
        """Test ConfigReloadRequested event creation."""
        event = ConfigReloadRequested()
        assert isinstance(event, Message)

    def test_config_changed_creation(self):
        """Test ConfigChanged event creation."""
        mock_config = MagicMock()
        event = ConfigChanged(config=mock_config)
        assert isinstance(event, Message)
        assert event.config == mock_config


class TestThemeEvents:
    """Test theme events."""

    def test_theme_change_requested_creation(self):
        """Test ThemeChangeRequested event creation."""
        event = ThemeChangeRequested(theme_mode="dark")
        assert isinstance(event, Message)
        assert event.theme_mode == "dark"

    def test_theme_changed_creation(self):
        """Test ThemeChanged event creation."""
        event = ThemeChanged(theme_mode="light")
        assert isinstance(event, Message)
        assert event.theme_mode == "light"


class TestEventIntegration:
    """Integration tests for event posting and handling."""

    @pytest.mark.asyncio
    async def test_event_posting_and_receiving(self):
        """Test that events can be posted and received in Textual app."""

        class TestWidget(Static):
            """Widget that captures events."""

            def __init__(self):
                super().__init__()
                self.events_received = []

            def on_mount(self) -> None:
                """Subscribe to events when mounted."""
                # In Textual, messages are delivered to all mounted widgets
                # that have a handler for that message type
                pass

            def on_bot_state_changed(self, event: BotStateChanged) -> None:
                """Handle bot state change event."""
                self.events_received.append(event)

        class TestApp(App):
            """Test app with widget."""

            def compose(self) -> ComposeResult:
                yield TestWidget()

        app = TestApp()
        async with app.run_test() as pilot:
            # Get widget - it's now mounted
            widget = app.query_one(TestWidget)

            # Post event to the app
            event = BotStateChanged(running=True, uptime=10.0)
            widget.post_message(event)

            # Wait for event processing
            await pilot.pause()

            # Verify event was received
            assert len(widget.events_received) == 1
            assert widget.events_received[0].running is True
            assert widget.events_received[0].uptime == 10.0

    @pytest.mark.skip(reason="Textual message routing between widgets is framework-specific")
    @pytest.mark.asyncio
    async def test_multiple_widgets_receive_same_event(self):
        """Test that multiple widgets can handle the same event.

        Note: Skipped because testing Textual's internal message routing
        is beyond the scope of our event system tests. The framework's
        message delivery is well-tested by Textual itself.
        """
        pass

    @pytest.mark.asyncio
    async def test_event_handler_invocation(self):
        """Test that event handlers are properly invoked."""

        class HandlerWidget(Static):
            """Widget that tracks handler invocations."""

            def __init__(self):
                super().__init__()
                self.handler_called = False
                self.event_data = None

            def on_bot_state_changed(self, event: BotStateChanged) -> None:
                """Handle the event and record it."""
                self.handler_called = True
                self.event_data = (event.running, event.uptime)

        class TestApp(App):
            """Test app."""

            def compose(self) -> ComposeResult:
                yield HandlerWidget()

        app = TestApp()
        async with app.run_test() as pilot:
            widget = app.query_one(HandlerWidget)

            # Post event
            widget.post_message(BotStateChanged(running=True, uptime=42.0))
            await pilot.pause()

            # Verify handler was called with correct data
            assert widget.handler_called is True
            assert widget.event_data == (True, 42.0)


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
            ValidationError,
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
        # Check a few key dataclass events have documented attributes
        assert "running" in BotStateChanged.__doc__
        assert "uptime" in BotStateChanged.__doc__

        assert "target_mode" in BotModeChangeRequested.__doc__

        assert "errors" in StateValidationFailed.__doc__
        assert "component" in StateValidationFailed.__doc__

"""Tests for state update/validation TUI events."""

from __future__ import annotations

from unittest.mock import MagicMock

from textual.message import Message

from gpt_trader.tui.events import (
    FieldValidationError,
    StateDeltaUpdateApplied,
    StateUpdateReceived,
    StateValidationFailed,
    StateValidationPassed,
)


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

    def test_field_validation_error_creation(self):
        """Test FieldValidationError event creation."""
        event = FieldValidationError(
            field="market_data", message="Price cannot be negative", severity="error", value=-10.0
        )
        assert isinstance(event, Message)
        assert event.field == "market_data"
        assert event.message == "Price cannot be negative"
        assert event.severity == "error"
        assert event.value == -10.0

    def test_field_validation_error_default_severity(self):
        """Test FieldValidationError with default severity."""
        event = FieldValidationError(field="test", message="Test error")
        assert event.severity == "error"
        assert event.value is None

    def test_state_validation_failed_creation(self):
        """Test StateValidationFailed event creation."""
        errors = [
            FieldValidationError(field="field1", message="Error 1"),
            FieldValidationError(field="field2", message="Error 2"),
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

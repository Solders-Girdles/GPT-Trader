"""Tests for ValidationIndicatorWidget."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gpt_trader.tui.events import FieldValidationError, StateValidationFailed, StateValidationPassed
from gpt_trader.tui.widgets.validation_indicator import ValidationIndicatorWidget


class TestValidationIndicatorWidget:
    """Test ValidationIndicatorWidget functionality."""

    def test_init_creates_empty_state(self):
        """Test initialization creates empty error state."""
        widget = ValidationIndicatorWidget()

        assert widget.error_count == 0
        assert widget.warning_count == 0
        assert widget.last_errors == []

    def test_on_state_validation_failed_updates_counts(self):
        """Test handling validation failure updates counts."""
        widget = ValidationIndicatorWidget()

        errors = [
            FieldValidationError(field="field1", message="error1", severity="error"),
            FieldValidationError(field="field2", message="warning1", severity="warning"),
        ]
        event = StateValidationFailed(errors=errors, component="test")

        widget.on_state_validation_failed(event)

        assert widget.error_count == 1
        assert widget.warning_count == 1
        assert len(widget.last_errors) == 2

    def test_on_state_validation_failed_stores_errors(self):
        """Test handling validation failure stores error details."""
        widget = ValidationIndicatorWidget()

        errors = [
            FieldValidationError(field="market.price", message="Invalid price"),
        ]
        event = StateValidationFailed(errors=errors, component="market")

        widget.on_state_validation_failed(event)

        assert widget.last_errors[0].field == "market.price"
        assert widget.last_errors[0].message == "Invalid price"

    def test_on_state_validation_passed_clears_counts(self):
        """Test handling validation success clears counts."""
        widget = ValidationIndicatorWidget()
        widget.error_count = 5
        widget.warning_count = 3

        # Patch set_timer to avoid needing event loop
        with patch.object(widget, "set_timer", return_value=MagicMock()):
            event = StateValidationPassed()
            widget.on_state_validation_passed(event)

        assert widget.error_count == 0
        assert widget.warning_count == 0
        assert widget.last_errors == []

    def test_format_error_message_single_error(self):
        """Test error message formatting with single error."""
        widget = ValidationIndicatorWidget()
        widget.error_count = 1
        widget.last_errors = [
            MagicMock(field="market.price", message="Negative price", severity="error")
        ]

        message = widget._format_error_message()

        assert "market.price" in message
        assert "Negative price" in message

    def test_format_error_message_multiple_errors(self):
        """Test error message formatting with multiple errors."""
        widget = ValidationIndicatorWidget()
        widget.error_count = 3
        widget.last_errors = []

        message = widget._format_error_message()

        assert "3" in message
        assert "error" in message

    def test_format_warning_message_single_warning(self):
        """Test warning message formatting with single warning."""
        widget = ValidationIndicatorWidget()
        widget.warning_count = 1
        widget.last_errors = [
            MagicMock(field="risk.leverage", message="High leverage", severity="warning")
        ]

        message = widget._format_warning_message()

        assert "risk.leverage" in message
        assert "High leverage" in message

    def test_get_validation_summary_with_errors(self):
        """Test summary string with errors."""
        widget = ValidationIndicatorWidget()
        widget.error_count = 2
        widget.warning_count = 1

        summary = widget.get_validation_summary()

        assert "2 errors" in summary
        assert "1 warnings" in summary

    def test_get_validation_summary_warnings_only(self):
        """Test summary string with warnings only."""
        widget = ValidationIndicatorWidget()
        widget.error_count = 0
        widget.warning_count = 3

        summary = widget.get_validation_summary()

        assert "3 warnings" in summary

    def test_get_validation_summary_valid(self):
        """Test summary string when valid."""
        widget = ValidationIndicatorWidget()
        widget.error_count = 0
        widget.warning_count = 0

        summary = widget.get_validation_summary()

        assert summary == "Valid"

    def test_get_error_details(self):
        """Test getting detailed error information."""
        widget = ValidationIndicatorWidget()
        widget.last_errors = [
            MagicMock(
                field="market.price",
                message="Negative price",
                severity="error",
                value="-100",
            )
        ]

        details = widget.get_error_details()

        assert len(details) == 1
        assert details[0]["field"] == "market.price"
        assert details[0]["message"] == "Negative price"
        assert details[0]["severity"] == "error"
        assert details[0]["value"] == "-100"

    def test_update_display_with_errors(self):
        """Test display update adds error class."""
        widget = ValidationIndicatorWidget()
        widget.error_count = 1
        widget.last_errors = [MagicMock(field="test", message="error", severity="error")]

        widget._update_display()

        assert "has-errors" in widget.classes

    def test_update_display_with_warnings(self):
        """Test display update adds warning class."""
        widget = ValidationIndicatorWidget()
        widget.error_count = 0
        widget.warning_count = 1
        widget.last_errors = [MagicMock(field="test", message="warning", severity="warning")]

        widget._update_display()

        assert "has-warnings" in widget.classes

    def test_update_display_valid(self):
        """Test display update adds valid class when no issues."""
        widget = ValidationIndicatorWidget()
        widget.error_count = 0
        widget.warning_count = 0

        widget._update_display()

        assert "valid" in widget.classes

"""Tests for ErrorIndicatorWidget."""

from unittest.mock import MagicMock

from gpt_trader.tui.widgets.error_indicator import ErrorEntry, ErrorIndicatorWidget


def test_error_entry_creation() -> None:
    """Test creating an ErrorEntry."""
    entry = ErrorEntry(
        widget="TestWidget", method="update_test", error="Test error", timestamp=1234.5
    )

    assert entry.widget == "TestWidget"
    assert entry.method == "update_test"
    assert entry.error == "Test error"
    assert entry.timestamp == 1234.5


def test_error_indicator_initialization() -> None:
    """Test error indicator initialization."""
    widget = ErrorIndicatorWidget(max_errors=10)

    assert widget.error_count == 0
    assert widget.is_collapsed is True
    assert widget._max_errors == 10
    assert len(widget._errors) == 0


def test_add_single_error() -> None:
    """Test adding a single error to the widget."""
    widget = ErrorIndicatorWidget(max_errors=10)

    widget.add_error("TestWidget", "failing_method", "Test error message")

    assert widget.error_count == 1
    assert len(widget._errors) == 1
    assert widget._errors[0].widget == "TestWidget"
    assert widget._errors[0].method == "failing_method"
    assert widget._errors[0].error == "Test error message"


def test_add_multiple_errors() -> None:
    """Test adding multiple errors to the widget."""
    widget = ErrorIndicatorWidget(max_errors=10)

    widget.add_error("Widget1", "method1", "Error 1")
    widget.add_error("Widget2", "method2", "Error 2")
    widget.add_error("Widget3", "method3", "Error 3")

    assert widget.error_count == 3
    assert len(widget._errors) == 3


def test_max_errors_fifo() -> None:
    """Test that max_errors limit is enforced with FIFO behavior."""
    widget = ErrorIndicatorWidget(max_errors=3)

    # Add 5 errors (more than max)
    widget.add_error("Widget1", "method1", "Error 1")
    widget.add_error("Widget2", "method2", "Error 2")
    widget.add_error("Widget3", "method3", "Error 3")
    widget.add_error("Widget4", "method4", "Error 4")
    widget.add_error("Widget5", "method5", "Error 5")

    # Should only have 3 errors (FIFO - oldest dropped)
    assert widget.error_count == 3
    assert len(widget._errors) == 3

    # First error should be dropped, should have errors 3, 4, 5
    assert widget._errors[0].widget == "Widget3"
    assert widget._errors[1].widget == "Widget4"
    assert widget._errors[2].widget == "Widget5"


def test_clear_errors() -> None:
    """Test clearing all errors."""
    widget = ErrorIndicatorWidget(max_errors=10)

    widget.add_error("Widget1", "method1", "Error 1")
    widget.add_error("Widget2", "method2", "Error 2")

    assert widget.error_count == 2

    widget.clear_errors()

    assert widget.error_count == 0
    assert len(widget._errors) == 0


def test_update_display_with_errors() -> None:
    """Test _update_display updates badge and visibility."""
    widget = ErrorIndicatorWidget(max_errors=10)

    # Mock query_one to return mock labels
    mock_badge = MagicMock()
    mock_list = MagicMock()

    def query_side_effect(selector, type=None):
        if selector == "#error-badge":
            return mock_badge
        else:  # error-list
            return mock_list

    widget.query_one = MagicMock(side_effect=query_side_effect)

    # Add errors
    widget.add_error("Widget1", "method1", "Error 1")
    widget._update_display()

    # Badge should be updated
    mock_badge.update.assert_called()
    # Check that update was called with error count (at least once during add_error flow)
    assert mock_badge.update.call_count >= 1


def test_update_display_multiple_errors() -> None:
    """Test _update_display with multiple errors."""
    widget = ErrorIndicatorWidget(max_errors=10)

    # Mock query_one
    mock_badge = MagicMock()
    mock_list = MagicMock()

    def query_side_effect(selector, type=None):
        if selector == "#error-badge":
            return mock_badge
        else:
            return mock_list

    widget.query_one = MagicMock(side_effect=query_side_effect)

    # Add multiple errors
    widget.add_error("Widget1", "method1", "Error 1")
    widget.add_error("Widget2", "method2", "Error 2")
    widget._update_display()

    # Badge should be updated
    assert mock_badge.update.call_count >= 2  # Once per add_error call


def test_update_display_no_errors() -> None:
    """Test _update_display with no errors."""
    widget = ErrorIndicatorWidget(max_errors=10)

    # Mock query_one
    mock_badge = MagicMock()
    mock_list = MagicMock()

    def query_side_effect(selector, type=None):
        if selector == "#error-badge":
            return mock_badge
        else:
            return mock_list

    widget.query_one = MagicMock(side_effect=query_side_effect)

    widget._update_display()

    # Badge should be updated
    mock_badge.update.assert_called()


def test_button_handler_toggle() -> None:
    """Test toggle button handler changes collapsed state."""
    widget = ErrorIndicatorWidget(max_errors=10)

    # Mock button event
    mock_button = MagicMock()
    mock_button.id = "toggle-btn"
    mock_button.label = "Expand"

    mock_event = MagicMock()
    mock_event.button = mock_button

    # Initially collapsed
    assert widget.is_collapsed is True

    # Press toggle
    widget.on_button_pressed(mock_event)

    # Should be expanded now
    assert widget.is_collapsed is False


def test_button_handler_clear() -> None:
    """Test clear button handler clears errors."""
    widget = ErrorIndicatorWidget(max_errors=10)

    # Add errors
    widget.add_error("Widget1", "method1", "Error 1")
    widget.add_error("Widget2", "method2", "Error 2")

    assert widget.error_count == 2

    # Mock button event
    mock_button = MagicMock()
    mock_button.id = "clear-btn"

    mock_event = MagicMock()
    mock_event.button = mock_button

    # Press clear
    widget.on_button_pressed(mock_event)

    # Errors should be cleared
    assert widget.error_count == 0
    assert len(widget._errors) == 0

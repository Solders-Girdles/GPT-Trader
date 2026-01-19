"""Tests for TUI status helper functions."""

from gpt_trader.tui.thresholds import (
    StatusLevel,
    get_status_class,
    get_status_color,
    get_status_icon,
)


class TestStatusLevel:
    """Tests for StatusLevel enum and helper functions."""

    def test_status_class_mapping(self):
        """Each status level maps to correct CSS class."""
        assert get_status_class(StatusLevel.OK) == "status-ok"
        assert get_status_class(StatusLevel.WARNING) == "status-warning"
        assert get_status_class(StatusLevel.CRITICAL) == "status-critical"

    def test_status_icon_mapping(self):
        """Each status level has an icon."""
        assert get_status_icon(StatusLevel.OK) == "✓"
        assert get_status_icon(StatusLevel.WARNING) == "⚠"
        assert get_status_icon(StatusLevel.CRITICAL) == "✗"

    def test_status_color_mapping(self):
        """Each status level has a Rich color."""
        assert get_status_color(StatusLevel.OK) == "green"
        assert get_status_color(StatusLevel.WARNING) == "yellow"
        assert get_status_color(StatusLevel.CRITICAL) == "red"

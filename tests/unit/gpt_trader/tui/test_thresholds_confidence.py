"""Tests for TUI confidence threshold functions."""

from gpt_trader.tui.thresholds import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    ConfidenceThresholds,
    StatusLevel,
    format_confidence_with_badge,
    get_confidence_label,
    get_confidence_status,
)


class TestConfidenceThresholds:
    """Tests for confidence threshold functions."""

    def test_default_thresholds_exist(self):
        """Default confidence thresholds are properly configured."""
        assert DEFAULT_CONFIDENCE_THRESHOLDS.confidence_low == 0.4
        assert DEFAULT_CONFIDENCE_THRESHOLDS.confidence_high == 0.7


class TestConfidenceStatus:
    """Tests for get_confidence_status function.

    Note: For confidence, higher is BETTER, so status mapping is inverted:
    - HIGH confidence (>= 0.7) = OK (green)
    - MEDIUM confidence (0.4-0.7) = WARNING (yellow)
    - LOW confidence (< 0.4) = CRITICAL (red)
    """

    def test_high_confidence_returns_ok(self):
        """High confidence (>= 0.7) returns OK status."""
        assert get_confidence_status(0.7) == StatusLevel.OK
        assert get_confidence_status(0.85) == StatusLevel.OK
        assert get_confidence_status(1.0) == StatusLevel.OK

    def test_medium_confidence_returns_warning(self):
        """Medium confidence (0.4-0.7) returns WARNING status."""
        assert get_confidence_status(0.4) == StatusLevel.WARNING
        assert get_confidence_status(0.5) == StatusLevel.WARNING
        assert get_confidence_status(0.69) == StatusLevel.WARNING

    def test_low_confidence_returns_critical(self):
        """Low confidence (< 0.4) returns CRITICAL status."""
        assert get_confidence_status(0.0) == StatusLevel.CRITICAL
        assert get_confidence_status(0.2) == StatusLevel.CRITICAL
        assert get_confidence_status(0.39) == StatusLevel.CRITICAL

    def test_custom_thresholds(self):
        """Custom confidence thresholds are respected."""
        custom = ConfidenceThresholds(confidence_low=0.5, confidence_high=0.8)
        assert get_confidence_status(0.45, custom) == StatusLevel.CRITICAL
        assert get_confidence_status(0.6, custom) == StatusLevel.WARNING
        assert get_confidence_status(0.85, custom) == StatusLevel.OK


class TestConfidenceLabel:
    """Tests for get_confidence_label function."""

    def test_ok_maps_to_high(self):
        """OK status maps to HIGH confidence label."""
        assert get_confidence_label(StatusLevel.OK) == "HIGH"

    def test_warning_maps_to_med(self):
        """WARNING status maps to MED confidence label."""
        assert get_confidence_label(StatusLevel.WARNING) == "MED"

    def test_critical_maps_to_low(self):
        """CRITICAL status maps to LOW confidence label."""
        assert get_confidence_label(StatusLevel.CRITICAL) == "LOW"


class TestFormatConfidenceWithBadge:
    """Tests for format_confidence_with_badge function."""

    def test_high_confidence_format(self):
        """High confidence formats with HIGH badge and ok class."""
        text, css_class = format_confidence_with_badge(0.85)
        assert text == "0.85 HIGH"
        assert css_class == "status-ok"

    def test_medium_confidence_format(self):
        """Medium confidence formats with MED badge and warning class."""
        text, css_class = format_confidence_with_badge(0.55)
        assert text == "0.55 MED"
        assert css_class == "status-warning"

    def test_low_confidence_format(self):
        """Low confidence formats with LOW badge and critical class."""
        text, css_class = format_confidence_with_badge(0.25)
        assert text == "0.25 LOW"
        assert css_class == "status-critical"

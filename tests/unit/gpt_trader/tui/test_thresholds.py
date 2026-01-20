"""Tests for TUI threshold functions: confidence, performance, and status helpers."""

from gpt_trader.tui.thresholds import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    ConfidenceThresholds,
    StatusLevel,
    format_confidence_with_badge,
    get_confidence_label,
    get_confidence_status,
    get_cpu_status,
    get_latency_status,
    get_memory_status,
    get_status_class,
    get_status_color,
    get_status_icon,
)


class TestConfidenceThresholds:
    """Tests for confidence threshold functions."""

    def test_default_thresholds_exist(self):
        """Default confidence thresholds are properly configured."""
        assert DEFAULT_CONFIDENCE_THRESHOLDS.confidence_low == 0.4
        assert DEFAULT_CONFIDENCE_THRESHOLDS.confidence_high == 0.7


class TestConfidenceStatus:
    """Tests for get_confidence_status function."""

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


class TestPerformanceThresholds:
    """Tests for performance metric thresholds."""

    def test_latency_ok(self):
        """Low latency returns OK status."""
        assert get_latency_status(30.0) == StatusLevel.OK

    def test_latency_warning(self):
        """Moderate latency returns WARNING status."""
        assert get_latency_status(100.0) == StatusLevel.WARNING

    def test_latency_critical(self):
        """High latency returns CRITICAL status."""
        assert get_latency_status(200.0) == StatusLevel.CRITICAL

    def test_cpu_ok(self):
        """Low CPU returns OK status."""
        assert get_cpu_status(30.0) == StatusLevel.OK

    def test_cpu_warning(self):
        """Moderate CPU returns WARNING status."""
        assert get_cpu_status(65.0) == StatusLevel.WARNING

    def test_cpu_critical(self):
        """High CPU returns CRITICAL status."""
        assert get_cpu_status(90.0) == StatusLevel.CRITICAL

    def test_memory_ok(self):
        """Low memory returns OK status."""
        assert get_memory_status(40.0) == StatusLevel.OK

    def test_memory_warning(self):
        """Moderate memory returns WARNING status."""
        assert get_memory_status(70.0) == StatusLevel.WARNING

    def test_memory_critical(self):
        """High memory returns CRITICAL status."""
        assert get_memory_status(90.0) == StatusLevel.CRITICAL


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

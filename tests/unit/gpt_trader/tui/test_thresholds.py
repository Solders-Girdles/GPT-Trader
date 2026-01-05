"""Tests for TUI threshold functions.

Covers both performance and risk threshold calculations with a focus on
verifying the abs() fix for loss ratio calculations.
"""


from gpt_trader.tui.thresholds import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    DEFAULT_RISK_THRESHOLDS,
    ConfidenceThresholds,
    RiskThresholds,
    StatusLevel,
    format_confidence_with_badge,
    get_confidence_label,
    get_confidence_status,
    get_cpu_status,
    get_latency_status,
    get_loss_ratio_status,
    get_memory_status,
    get_risk_score_status,
    get_risk_status_label,
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


class TestRiskThresholds:
    """Tests for risk metric thresholds."""

    def test_default_thresholds_exist(self):
        """Default risk thresholds are properly configured."""
        assert DEFAULT_RISK_THRESHOLDS.loss_ratio_ok == 0.50
        assert DEFAULT_RISK_THRESHOLDS.loss_ratio_warn == 0.75
        assert DEFAULT_RISK_THRESHOLDS.risk_score_ok == 2
        assert DEFAULT_RISK_THRESHOLDS.risk_score_warn == 5


class TestLossRatioStatus:
    """Tests for get_loss_ratio_status function.

    CRITICAL: These tests verify the abs() fix that prevents the bug where
    negative losses (actual losses) incorrectly showed green status.
    """

    def test_no_limit_configured_returns_ok(self):
        """When no loss limit is set, always returns OK."""
        assert get_loss_ratio_status(-0.05, 0.0) == StatusLevel.OK
        assert get_loss_ratio_status(-0.10, 0.0) == StatusLevel.OK
        assert get_loss_ratio_status(0.05, 0.0) == StatusLevel.OK

    def test_zero_loss_returns_ok(self):
        """Zero loss is within limits."""
        assert get_loss_ratio_status(0.0, 0.10) == StatusLevel.OK

    def test_small_loss_returns_ok(self):
        """Loss under 50% of limit returns OK."""
        # -3% loss against 10% limit = 30% utilization = OK
        assert get_loss_ratio_status(-0.03, 0.10) == StatusLevel.OK

    def test_moderate_loss_returns_warning(self):
        """Loss between 50-75% of limit returns WARNING."""
        # -6% loss against 10% limit = 60% utilization = WARNING
        assert get_loss_ratio_status(-0.06, 0.10) == StatusLevel.WARNING

    def test_high_loss_returns_critical(self):
        """Loss over 75% of limit returns CRITICAL."""
        # -8% loss against 10% limit = 80% utilization = CRITICAL
        assert get_loss_ratio_status(-0.08, 0.10) == StatusLevel.CRITICAL

    def test_loss_at_limit_returns_critical(self):
        """Loss at 100% of limit returns CRITICAL."""
        assert get_loss_ratio_status(-0.10, 0.10) == StatusLevel.CRITICAL

    def test_loss_exceeds_limit_returns_critical(self):
        """Loss exceeding limit returns CRITICAL."""
        assert get_loss_ratio_status(-0.15, 0.10) == StatusLevel.CRITICAL

    def test_abs_fix_negative_loss_handled_correctly(self):
        """CRITICAL: Negative losses are converted to positive for ratio.

        This is the key test for the abs() fix. Before the fix:
        - loss_ratio = -0.05 / 0.10 = -0.50
        - -0.50 < 0.50 (threshold) = TRUE = OK (WRONG!)

        After the fix:
        - loss_ratio = abs(-0.05) / 0.10 = 0.50
        - 0.50 < 0.50 = FALSE, 0.50 < 0.75 = TRUE = WARNING (CORRECT!)
        """
        # At exactly 50% - should be WARNING not OK
        assert get_loss_ratio_status(-0.05, 0.10) == StatusLevel.WARNING

    def test_profit_does_not_affect_risk(self):
        """Positive P&L (profit) treated same as loss for risk calculation.

        Profit is still "distance from zero" for risk purposes.
        """
        # +5% profit against 10% limit = 50% utilization = WARNING
        # This is correct behavior - profit doesn't reduce risk utilization
        assert get_loss_ratio_status(0.05, 0.10) == StatusLevel.WARNING

    def test_boundary_at_ok_threshold(self):
        """Exactly at OK threshold boundary."""
        # 49.9% utilization = OK
        assert get_loss_ratio_status(-0.0499, 0.10) == StatusLevel.OK
        # 50% utilization = WARNING (at boundary)
        assert get_loss_ratio_status(-0.05, 0.10) == StatusLevel.WARNING

    def test_boundary_at_warn_threshold(self):
        """Around WARNING threshold boundary."""
        # 74% utilization = WARNING
        assert get_loss_ratio_status(-0.074, 0.10) == StatusLevel.WARNING
        # 76% utilization = CRITICAL (above threshold)
        assert get_loss_ratio_status(-0.076, 0.10) == StatusLevel.CRITICAL

    def test_custom_thresholds(self):
        """Custom thresholds are respected."""
        custom = RiskThresholds(
            loss_ratio_ok=0.30,  # Tighter: 30%
            loss_ratio_warn=0.60,  # Tighter: 60%
        )
        # 25% = OK with custom thresholds
        assert get_loss_ratio_status(-0.025, 0.10, custom) == StatusLevel.OK
        # 40% = WARNING with custom thresholds (would be OK with defaults)
        assert get_loss_ratio_status(-0.04, 0.10, custom) == StatusLevel.WARNING
        # 70% = CRITICAL with custom thresholds
        assert get_loss_ratio_status(-0.07, 0.10, custom) == StatusLevel.CRITICAL


class TestRiskScoreStatus:
    """Tests for get_risk_score_status function."""

    def test_low_score_returns_ok(self):
        """Score below OK threshold is LOW risk."""
        assert get_risk_score_status(0) == StatusLevel.OK
        assert get_risk_score_status(1) == StatusLevel.OK

    def test_medium_score_returns_warning(self):
        """Score between thresholds is MEDIUM risk."""
        assert get_risk_score_status(2) == StatusLevel.WARNING
        assert get_risk_score_status(3) == StatusLevel.WARNING
        assert get_risk_score_status(4) == StatusLevel.WARNING

    def test_high_score_returns_critical(self):
        """Score at or above warning threshold is HIGH risk."""
        assert get_risk_score_status(5) == StatusLevel.CRITICAL
        assert get_risk_score_status(10) == StatusLevel.CRITICAL

    def test_custom_thresholds(self):
        """Custom score thresholds are respected."""
        custom = RiskThresholds(risk_score_ok=1, risk_score_warn=3)
        assert get_risk_score_status(0, custom) == StatusLevel.OK
        assert get_risk_score_status(1, custom) == StatusLevel.WARNING
        assert get_risk_score_status(3, custom) == StatusLevel.CRITICAL


class TestRiskStatusLabel:
    """Tests for get_risk_status_label function."""

    def test_ok_maps_to_low(self):
        """OK status maps to LOW risk label."""
        assert get_risk_status_label(StatusLevel.OK) == "LOW"

    def test_warning_maps_to_medium(self):
        """WARNING status maps to MEDIUM risk label."""
        assert get_risk_status_label(StatusLevel.WARNING) == "MEDIUM"

    def test_critical_maps_to_high(self):
        """CRITICAL status maps to HIGH risk label."""
        assert get_risk_status_label(StatusLevel.CRITICAL) == "HIGH"


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
        # Tighter thresholds
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

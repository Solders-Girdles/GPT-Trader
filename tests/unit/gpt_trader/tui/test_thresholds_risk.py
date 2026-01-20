"""Tests for TUI risk threshold calculations: loss-ratio, risk-score, and order status."""

from gpt_trader.tui.thresholds import (
    DEFAULT_RISK_THRESHOLDS,
    RiskThresholds,
    StatusLevel,
    get_loss_ratio_status,
    get_order_status_level,
    get_risk_score_status,
    get_risk_status_label,
)


class TestRiskThresholds:
    """Tests for risk metric thresholds."""

    def test_default_thresholds_exist(self):
        """Default risk thresholds are properly configured."""
        assert DEFAULT_RISK_THRESHOLDS.loss_ratio_ok == 0.50
        assert DEFAULT_RISK_THRESHOLDS.loss_ratio_warn == 0.75
        assert DEFAULT_RISK_THRESHOLDS.risk_score_ok == 2
        assert DEFAULT_RISK_THRESHOLDS.risk_score_warn == 5


class TestLossRatioStatus:
    """Tests for get_loss_ratio_status function."""

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
        assert get_loss_ratio_status(-0.03, 0.10) == StatusLevel.OK

    def test_moderate_loss_returns_warning(self):
        """Loss between 50-75% of limit returns WARNING."""
        assert get_loss_ratio_status(-0.06, 0.10) == StatusLevel.WARNING

    def test_high_loss_returns_critical(self):
        """Loss over 75% of limit returns CRITICAL."""
        assert get_loss_ratio_status(-0.08, 0.10) == StatusLevel.CRITICAL

    def test_loss_at_limit_returns_critical(self):
        """Loss at 100% of limit returns CRITICAL."""
        assert get_loss_ratio_status(-0.10, 0.10) == StatusLevel.CRITICAL

    def test_loss_exceeds_limit_returns_critical(self):
        """Loss exceeding limit returns CRITICAL."""
        assert get_loss_ratio_status(-0.15, 0.10) == StatusLevel.CRITICAL

    def test_abs_fix_negative_loss_handled_correctly(self):
        """Negative losses are converted to positive for ratio calculation."""
        assert get_loss_ratio_status(-0.05, 0.10) == StatusLevel.WARNING

    def test_profit_does_not_affect_risk(self):
        """Positive P&L (profit) treated same as loss for risk calculation."""
        assert get_loss_ratio_status(0.05, 0.10) == StatusLevel.WARNING

    def test_boundary_at_ok_threshold(self):
        """Exactly at OK threshold boundary."""
        assert get_loss_ratio_status(-0.0499, 0.10) == StatusLevel.OK
        assert get_loss_ratio_status(-0.05, 0.10) == StatusLevel.WARNING

    def test_boundary_at_warn_threshold(self):
        """Around WARNING threshold boundary."""
        assert get_loss_ratio_status(-0.074, 0.10) == StatusLevel.WARNING
        assert get_loss_ratio_status(-0.076, 0.10) == StatusLevel.CRITICAL

    def test_custom_thresholds(self):
        """Custom thresholds are respected."""
        custom = RiskThresholds(loss_ratio_ok=0.30, loss_ratio_warn=0.60)
        assert get_loss_ratio_status(-0.025, 0.10, custom) == StatusLevel.OK
        assert get_loss_ratio_status(-0.04, 0.10, custom) == StatusLevel.WARNING
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


class TestOrderStatusLevel:
    """Tests for get_order_status_level function."""

    def test_open_status_returns_ok(self):
        """OPEN orders are normal - OK status."""
        assert get_order_status_level("OPEN") == StatusLevel.OK
        assert get_order_status_level("open") == StatusLevel.OK

    def test_pending_status_returns_ok(self):
        """PENDING orders are normal - OK status."""
        assert get_order_status_level("PENDING") == StatusLevel.OK

    def test_filled_status_returns_ok(self):
        """FILLED orders are complete - OK status."""
        assert get_order_status_level("FILLED") == StatusLevel.OK

    def test_cancelled_status_returns_ok(self):
        """CANCELLED orders are user-initiated - OK status (not error)."""
        assert get_order_status_level("CANCELLED") == StatusLevel.OK
        assert get_order_status_level("CANCELED") == StatusLevel.OK

    def test_partial_status_returns_warning(self):
        """PARTIAL fills need attention - WARNING status."""
        assert get_order_status_level("PARTIAL") == StatusLevel.WARNING

    def test_expired_status_returns_warning(self):
        """EXPIRED orders may indicate stale strategy - WARNING status."""
        assert get_order_status_level("EXPIRED") == StatusLevel.WARNING

    def test_rejected_status_returns_critical(self):
        """REJECTED orders are errors - CRITICAL status."""
        assert get_order_status_level("REJECTED") == StatusLevel.CRITICAL

    def test_failed_status_returns_critical(self):
        """FAILED orders are errors - CRITICAL status."""
        assert get_order_status_level("FAILED") == StatusLevel.CRITICAL

    def test_unknown_status_returns_ok(self):
        """Unknown statuses default to OK (safe fallback)."""
        assert get_order_status_level("UNKNOWN") == StatusLevel.OK
        assert get_order_status_level("SOME_NEW_STATUS") == StatusLevel.OK

    def test_case_insensitive(self):
        """Status matching is case-insensitive."""
        assert get_order_status_level("rejected") == StatusLevel.CRITICAL
        assert get_order_status_level("Rejected") == StatusLevel.CRITICAL
        assert get_order_status_level("REJECTED") == StatusLevel.CRITICAL

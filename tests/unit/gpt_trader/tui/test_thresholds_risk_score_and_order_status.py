"""Tests for TUI risk-score and order-status thresholds."""

from gpt_trader.tui.thresholds import (
    RiskThresholds,
    StatusLevel,
    get_order_status_level,
    get_risk_score_status,
    get_risk_status_label,
)


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
    """Tests for get_order_status_level function.

    Maps order status strings to visual severity for at-a-glance readability:
    - OK (green): OPEN, PENDING, FILLED, CANCELLED
    - WARNING (yellow): PARTIAL, EXPIRED
    - CRITICAL (red): REJECTED, FAILED
    """

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
        assert get_order_status_level("CANCELED") == StatusLevel.OK  # US spelling

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

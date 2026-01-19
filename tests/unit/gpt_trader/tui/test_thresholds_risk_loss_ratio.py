"""Tests for TUI risk loss-ratio threshold calculations."""

from gpt_trader.tui.thresholds import (
    DEFAULT_RISK_THRESHOLDS,
    RiskThresholds,
    StatusLevel,
    get_loss_ratio_status,
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

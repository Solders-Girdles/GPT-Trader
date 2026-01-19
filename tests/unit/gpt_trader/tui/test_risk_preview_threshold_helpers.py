"""Tests for risk_preview threshold helper functions."""

from gpt_trader.tui.risk_preview import _get_guard_impacts, _get_ratio_status
from gpt_trader.tui.thresholds import DEFAULT_RISK_THRESHOLDS, StatusLevel


class TestGetRatioStatus:
    """Tests for _get_ratio_status helper."""

    def test_status_boundaries(self):
        """Status transitions at correct threshold boundaries."""
        # OK: below 50%
        assert _get_ratio_status(0.0, DEFAULT_RISK_THRESHOLDS) == StatusLevel.OK
        assert _get_ratio_status(0.49, DEFAULT_RISK_THRESHOLDS) == StatusLevel.OK
        # WARNING: 50-75% (inclusive at 50, exclusive at 75)
        assert _get_ratio_status(0.50, DEFAULT_RISK_THRESHOLDS) == StatusLevel.WARNING
        assert _get_ratio_status(0.74, DEFAULT_RISK_THRESHOLDS) == StatusLevel.WARNING
        # CRITICAL: 75%+
        assert _get_ratio_status(0.75, DEFAULT_RISK_THRESHOLDS) == StatusLevel.CRITICAL
        assert _get_ratio_status(1.5, DEFAULT_RISK_THRESHOLDS) == StatusLevel.CRITICAL


class TestGetGuardImpacts:
    """Tests for _get_guard_impacts helper."""

    def test_guards_by_threshold(self):
        """Guards trigger at correct thresholds with reasons."""
        # Below warning: no guards
        assert _get_guard_impacts(0.74, DEFAULT_RISK_THRESHOLDS) == []

        # At warning: DailyLossGuard triggers
        impacts = _get_guard_impacts(0.75, DEFAULT_RISK_THRESHOLDS)
        assert len(impacts) == 1
        assert impacts[0].name == "DailyLossGuard"
        assert "75%" in impacts[0].reason

        # At 100%: both guards trigger
        impacts = _get_guard_impacts(1.0, DEFAULT_RISK_THRESHOLDS)
        guard_names = [g.name for g in impacts]
        assert "DailyLossGuard" in guard_names
        assert "ReduceOnlyMode" in guard_names

    def test_reasons_include_threshold_text(self):
        """Guard reasons include threshold percentage text."""
        impacts = _get_guard_impacts(1.0, DEFAULT_RISK_THRESHOLDS)
        reasons = {g.name: g.reason for g in impacts}

        assert ">=75% of limit" in reasons["DailyLossGuard"]
        assert ">=100% of limit" in reasons["ReduceOnlyMode"]

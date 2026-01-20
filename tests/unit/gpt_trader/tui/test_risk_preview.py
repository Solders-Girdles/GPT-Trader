"""Tests for risk_preview dataclass models and threshold helper functions."""

import pytest

from gpt_trader.tui.risk_preview import (
    GuardImpact,
    GuardPositionImpact,
    RiskPreviewResult,
    RiskPreviewScenario,
    _get_guard_impacts,
    _get_ratio_status,
)
from gpt_trader.tui.thresholds import DEFAULT_RISK_THRESHOLDS, StatusLevel


class TestRiskPreviewScenario:
    """Tests for RiskPreviewScenario dataclass."""

    def test_scenario_creation_and_immutability(self):
        """Scenario stores label and shock percentage, is immutable."""
        scenario = RiskPreviewScenario(label="+5%", shock_pct=0.05)
        assert scenario.label == "+5%"
        assert scenario.shock_pct == 0.05
        with pytest.raises(AttributeError):
            scenario.label = "changed"  # type: ignore[misc]


class TestRiskPreviewResult:
    """Tests for RiskPreviewResult dataclass."""

    def test_result_creation(self):
        """Result stores all preview fields with defaults."""
        result = RiskPreviewResult(
            label="-5%",
            projected_loss_pct=75.0,
            status=StatusLevel.WARNING,
            guard_impacts=[GuardImpact(name="DailyLossGuard", reason=">=75% of limit")],
        )
        assert result.label == "-5%"
        assert result.projected_loss_pct == 75.0
        assert len(result.guard_impacts) == 1
        assert result.guard_impacts[0].name == "DailyLossGuard"
        assert result.guard_impacts[0].reason == ">=75% of limit"

        # Test default empty guards
        result2 = RiskPreviewResult(label="+2%", projected_loss_pct=30.0, status=StatusLevel.OK)
        assert result2.guard_impacts == []


class TestGuardPositionImpact:
    """Tests for GuardPositionImpact dataclass."""

    def test_guard_position_impact_dataclass(self):
        """GuardPositionImpact stores all fields correctly."""
        impact = GuardPositionImpact(
            guard_name="DailyLossGuard",
            symbol="BTC-USD",
            current_pnl_pct=-2.0,
            projected_pnl_pct=-6.8,
            limit_pct=5.0,
            reason="-2.0% → -6.8%",
        )
        assert impact.guard_name == "DailyLossGuard"
        assert impact.symbol == "BTC-USD"
        assert impact.current_pnl_pct == -2.0
        assert impact.projected_pnl_pct == -6.8
        assert impact.limit_pct == 5.0
        assert "-2.0% → -6.8%" in impact.reason


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

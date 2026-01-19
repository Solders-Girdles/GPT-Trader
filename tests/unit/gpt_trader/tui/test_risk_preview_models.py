"""Tests for risk_preview dataclass models."""

import pytest

from gpt_trader.tui.risk_preview import (
    GuardImpact,
    GuardPositionImpact,
    RiskPreviewResult,
    RiskPreviewScenario,
)
from gpt_trader.tui.thresholds import StatusLevel


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

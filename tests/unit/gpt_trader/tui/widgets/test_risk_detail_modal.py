"""Tests for RiskDetailModal."""

import time

from gpt_trader.tui.types import RiskGuard, RiskState
from gpt_trader.tui.widgets.risk_detail_modal import RiskDetailModal


class TestRiskDetailModal:
    """Tests for RiskDetailModal risk score calculation and formatting."""

    def test_risk_score_no_factors(self):
        """Zero score when no risk factors present."""
        data = RiskState(
            max_leverage=1.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=0.0,
            reduce_only_mode=False,
            active_guards=[],
        )
        modal = RiskDetailModal(data)
        assert modal._calculate_risk_score(data) == 0

    def test_risk_score_with_moderate_loss(self):
        """Score increases with loss 25-50% of limit (+1)."""
        data = RiskState(
            max_leverage=1.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.03,  # 30% utilization
            reduce_only_mode=False,
            active_guards=[],
        )
        modal = RiskDetailModal(data)
        assert modal._calculate_risk_score(data) == 1

    def test_risk_score_with_warning_loss(self):
        """Score increases with loss 50-75% of limit (+2)."""
        data = RiskState(
            max_leverage=1.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.06,  # 60% utilization
            reduce_only_mode=False,
            active_guards=[],
        )
        modal = RiskDetailModal(data)
        assert modal._calculate_risk_score(data) == 2

    def test_risk_score_with_critical_loss(self):
        """Score increases with loss >75% of limit (+3)."""
        data = RiskState(
            max_leverage=1.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.08,  # 80% utilization
            reduce_only_mode=False,
            active_guards=[],
        )
        modal = RiskDetailModal(data)
        assert modal._calculate_risk_score(data) == 3

    def test_risk_score_with_reduce_only(self):
        """Reduce-only mode adds +3 to score."""
        data = RiskState(
            max_leverage=1.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=0.0,
            reduce_only_mode=True,
            reduce_only_reason="Risk limit",
            active_guards=[],
        )
        modal = RiskDetailModal(data)
        assert modal._calculate_risk_score(data) == 3

    def test_risk_score_with_one_guard(self):
        """One guard adds +1 to score."""
        data = RiskState(
            max_leverage=1.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=0.0,
            reduce_only_mode=False,
            active_guards=["MaxDrawdown"],
        )
        modal = RiskDetailModal(data)
        assert modal._calculate_risk_score(data) == 1

    def test_risk_score_with_two_guards(self):
        """Two guards still adds +1 to score."""
        data = RiskState(
            max_leverage=1.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=0.0,
            reduce_only_mode=False,
            active_guards=["MaxDrawdown", "DailyLossLimit"],
        )
        modal = RiskDetailModal(data)
        assert modal._calculate_risk_score(data) == 1

    def test_risk_score_with_three_or_more_guards(self):
        """Three or more guards adds +2 to score."""
        data = RiskState(
            max_leverage=1.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=0.0,
            reduce_only_mode=False,
            active_guards=["MaxDrawdown", "DailyLossLimit", "VolatilityGuard"],
        )
        modal = RiskDetailModal(data)
        assert modal._calculate_risk_score(data) == 2

    def test_risk_score_cumulative(self):
        """All factors combine cumulatively."""
        data = RiskState(
            max_leverage=1.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.08,  # +3 (critical)
            reduce_only_mode=True,  # +3
            active_guards=["MaxDrawdown", "DailyLossLimit", "VolatilityGuard"],  # +2
        )
        modal = RiskDetailModal(data)
        # 3 (critical loss) + 3 (reduce-only) + 2 (3 guards) = 8
        assert modal._calculate_risk_score(data) == 8

    def test_score_breakdown_no_factors(self):
        """Breakdown shows 'no risk factors' when empty."""
        data = RiskState(
            max_leverage=1.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=0.0,
            reduce_only_mode=False,
            active_guards=[],
        )
        modal = RiskDetailModal(data)
        breakdown = modal._format_score_breakdown(data)
        assert breakdown == "No risk factors detected"

    def test_score_breakdown_with_factors(self):
        """Breakdown lists contributing factors."""
        data = RiskState(
            max_leverage=1.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.08,  # critical loss
            reduce_only_mode=True,
            active_guards=["MaxDrawdown"],
        )
        modal = RiskDetailModal(data)
        breakdown = modal._format_score_breakdown(data)
        assert "Loss >75% of limit: +3" in breakdown
        assert "Reduce-only active: +3" in breakdown
        assert "1 guard(s) active: +1" in breakdown

    def test_score_breakdown_guards_three_plus(self):
        """Breakdown shows +2 for 3+ guards."""
        data = RiskState(
            max_leverage=1.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=0.0,
            reduce_only_mode=False,
            active_guards=["A", "B", "C"],
        )
        modal = RiskDetailModal(data)
        breakdown = modal._format_score_breakdown(data)
        assert "3 guards active: +2" in breakdown


class TestEnhancedGuardVisibility:
    """Tests for enhanced guard visibility features."""

    def test_get_sorted_guards_by_severity(self):
        """Guards are sorted by severity (highest first)."""
        guards = [
            RiskGuard(name="LowGuard", severity="LOW"),
            RiskGuard(name="CriticalGuard", severity="CRITICAL"),
            RiskGuard(name="MediumGuard", severity="MEDIUM"),
            RiskGuard(name="HighGuard", severity="HIGH"),
        ]
        data = RiskState(guards=guards)
        modal = RiskDetailModal(data)

        sorted_guards = modal._get_sorted_guards(data)
        assert len(sorted_guards) == 4
        assert sorted_guards[0].name == "CriticalGuard"
        assert sorted_guards[1].name == "HighGuard"
        assert sorted_guards[2].name == "MediumGuard"
        assert sorted_guards[3].name == "LowGuard"

    def test_get_sorted_guards_fallback_to_legacy(self):
        """Falls back to legacy active_guards when no enhanced guards."""
        data = RiskState(
            active_guards=["DailyLossGuard", "PositionSizeLimit"],
            guards=[],  # No enhanced guards
        )
        modal = RiskDetailModal(data)

        sorted_guards = modal._get_sorted_guards(data)
        assert len(sorted_guards) == 2
        # Should be converted to RiskGuard objects
        assert all(isinstance(g, RiskGuard) for g in sorted_guards)

    def test_infer_severity_critical(self):
        """Drawdown/loss guards inferred as CRITICAL."""
        data = RiskState()
        modal = RiskDetailModal(data)

        assert modal._infer_severity("DailyLossGuard") == "CRITICAL"
        assert modal._infer_severity("MaxDrawdownLimit") == "CRITICAL"
        assert modal._infer_severity("MarginCallGuard") == "CRITICAL"

    def test_infer_severity_high(self):
        """Volatility/exposure guards inferred as HIGH."""
        data = RiskState()
        modal = RiskDetailModal(data)

        assert modal._infer_severity("VolatilityGuard") == "HIGH"
        assert modal._infer_severity("ExposureLimit") == "HIGH"
        assert modal._infer_severity("LeverageGuard") == "HIGH"

    def test_infer_severity_medium(self):
        """Position/size guards inferred as MEDIUM."""
        data = RiskState()
        modal = RiskDetailModal(data)

        assert modal._infer_severity("PositionSizeLimit") == "MEDIUM"
        assert modal._infer_severity("DailySizeGuard") == "MEDIUM"

    def test_infer_severity_default_low(self):
        """Unknown guards default to LOW."""
        data = RiskState()
        modal = RiskDetailModal(data)

        assert modal._infer_severity("RandomGuard") == "LOW"
        assert modal._infer_severity("SomeOtherThing") == "LOW"

    def test_format_guard_row_with_recent_trigger(self):
        """Guard row shows recent trigger in red."""
        now = time.time()
        guard = RiskGuard(
            name="TestGuard",
            severity="HIGH",
            last_triggered=now - 30,  # 30 seconds ago
            triggered_count=5,
        )
        data = RiskState()
        modal = RiskDetailModal(data)

        formatted = modal._format_guard_row(guard)
        text_str = str(formatted)
        assert "TestGuard" in text_str
        assert "30s ago" in text_str or "29s ago" in text_str or "31s ago" in text_str
        assert "×5" in text_str  # Trigger count

    def test_format_guard_row_never_triggered(self):
        """Guard row shows 'never' for untriggered guards."""
        guard = RiskGuard(
            name="TestGuard",
            severity="MEDIUM",
            last_triggered=0.0,  # Never triggered
            triggered_count=0,
        )
        data = RiskState()
        modal = RiskDetailModal(data)

        formatted = modal._format_guard_row(guard)
        text_str = str(formatted)
        assert "TestGuard" in text_str
        assert "never" in text_str

    def test_format_age_seconds(self):
        """Age formatting for seconds."""
        data = RiskState()
        modal = RiskDetailModal(data)

        now = time.time()
        assert modal._format_age(now - 45) == "45s ago"

    def test_format_age_minutes(self):
        """Age formatting for minutes."""
        data = RiskState()
        modal = RiskDetailModal(data)

        now = time.time()
        assert modal._format_age(now - 180) == "3m ago"

    def test_format_age_hours(self):
        """Age formatting for hours."""
        data = RiskState()
        modal = RiskDetailModal(data)

        now = time.time()
        assert modal._format_age(now - 7200) == "2h ago"

    def test_format_age_days(self):
        """Age formatting for days."""
        data = RiskState()
        modal = RiskDetailModal(data)

        now = time.time()
        assert modal._format_age(now - 172800) == "2d ago"

    def test_severity_order_property(self):
        """RiskGuard.severity_order returns correct numeric values."""
        assert RiskGuard(name="test", severity="LOW").severity_order == 1
        assert RiskGuard(name="test", severity="MEDIUM").severity_order == 2
        assert RiskGuard(name="test", severity="HIGH").severity_order == 3
        assert RiskGuard(name="test", severity="CRITICAL").severity_order == 4

    def test_severity_order_case_insensitive(self):
        """Severity order works regardless of case."""
        assert RiskGuard(name="test", severity="low").severity_order == 1
        assert RiskGuard(name="test", severity="HIGH").severity_order == 3
        assert RiskGuard(name="test", severity="Critical").severity_order == 4

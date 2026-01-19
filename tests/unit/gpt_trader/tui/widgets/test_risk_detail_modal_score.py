"""Tests for RiskDetailModal risk score calculation and formatting."""

from gpt_trader.tui.types import RiskGuard, RiskState
from gpt_trader.tui.widgets.risk_detail_modal import RiskDetailModal


class TestRiskDetailModalScore:
    """Tests for RiskDetailModal risk score calculation and formatting."""

    def test_risk_score_no_factors(self):
        """Zero score when no risk factors present."""
        data = RiskState(
            max_leverage=1.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=0.0,
            reduce_only_mode=False,
            guards=[],
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
            guards=[],
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
            guards=[],
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
            guards=[],
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
            guards=[],
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
            guards=[RiskGuard(name="MaxDrawdown")],
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
            guards=[
                RiskGuard(name="MaxDrawdown"),
                RiskGuard(name="DailyLossLimit"),
            ],
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
            guards=[
                RiskGuard(name="MaxDrawdown"),
                RiskGuard(name="DailyLossLimit"),
                RiskGuard(name="VolatilityGuard"),
            ],
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
            guards=[
                RiskGuard(name="MaxDrawdown"),
                RiskGuard(name="DailyLossLimit"),
                RiskGuard(name="VolatilityGuard"),
            ],  # +2
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
            guards=[],
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
            guards=[RiskGuard(name="MaxDrawdown")],
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
            guards=[
                RiskGuard(name="A"),
                RiskGuard(name="B"),
                RiskGuard(name="C"),
            ],
        )
        modal = RiskDetailModal(data)
        breakdown = modal._format_score_breakdown(data)
        assert "3 guards active: +2" in breakdown

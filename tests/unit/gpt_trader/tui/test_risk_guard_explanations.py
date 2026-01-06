"""Tests for risk guard explanation helper."""

from gpt_trader.tui.risk_guard_explanations import (
    get_guard_explanation,
    get_guard_explanations,
)
from gpt_trader.tui.types import RiskState


class TestDailyLossGuardExplanation:
    """Tests for daily loss guard explanations."""

    def test_daily_loss_guard_with_usage(self) -> None:
        """Test daily loss guard shows usage percentage."""
        state = RiskState(
            daily_loss_limit_pct=0.05,  # 5% limit
            current_daily_loss_pct=-0.031,  # 3.1% loss (62% of limit)
        )
        explanation = get_guard_explanation("DailyLossGuard", state)

        assert "triggers at >= 75%" in explanation
        assert "current 62%" in explanation

    def test_daily_loss_guard_no_limit_configured(self) -> None:
        """Test daily loss guard when limit is not configured."""
        state = RiskState(daily_loss_limit_pct=0.0)
        explanation = get_guard_explanation("DailyLossGuard", state)

        assert "limit not configured" in explanation

    def test_daily_loss_guard_alternative_names(self) -> None:
        """Test daily loss guard matches alternative naming patterns."""
        state = RiskState(daily_loss_limit_pct=0.05, current_daily_loss_pct=-0.025)

        # Various naming patterns should all match
        for name in ["DailyLossGuard", "daily_loss_guard", "LossLimitGuard"]:
            explanation = get_guard_explanation(name, state)
            assert "triggers at" in explanation


class TestLeverageGuardExplanation:
    """Tests for leverage guard explanations."""

    def test_leverage_guard_shows_threshold(self) -> None:
        """Test leverage guard shows max leverage threshold."""
        state = RiskState(max_leverage=5.0)
        explanation = get_guard_explanation("MaxLeverageGuard", state)

        assert "triggers at >= 5.0x" in explanation

    def test_leverage_guard_no_limit(self) -> None:
        """Test leverage guard when limit is not configured."""
        state = RiskState(max_leverage=0.0)
        explanation = get_guard_explanation("LeverageGuard", state)

        assert "limit not configured" in explanation


class TestReduceOnlyGuardExplanation:
    """Tests for reduce-only guard explanations."""

    def test_reduce_only_active_with_reason(self) -> None:
        """Test reduce-only guard when active shows reason."""
        state = RiskState(
            reduce_only_mode=True,
            reduce_only_reason="daily_loss_exceeded",
        )
        explanation = get_guard_explanation("ReduceOnlyGuard", state)

        assert "active" in explanation
        assert "daily_loss_exceeded" in explanation

    def test_reduce_only_active_no_reason(self) -> None:
        """Test reduce-only guard when active without specific reason."""
        state = RiskState(reduce_only_mode=True, reduce_only_reason="")
        explanation = get_guard_explanation("reduce_only_mode", state)

        assert "manually enabled" in explanation

    def test_reduce_only_inactive(self) -> None:
        """Test reduce-only guard when not active."""
        state = RiskState(reduce_only_mode=False)
        explanation = get_guard_explanation("ReduceOnlyGuard", state)

        assert "triggers when reduce-only mode is enabled" in explanation


class TestOtherGuardExplanations:
    """Tests for other guard type explanations."""

    def test_position_limit_guard(self) -> None:
        """Test position limit guard explanation."""
        state = RiskState()
        explanation = get_guard_explanation("PositionLimitGuard", state)

        assert "position count" in explanation

    def test_drawdown_guard(self) -> None:
        """Test drawdown guard explanation."""
        state = RiskState()
        explanation = get_guard_explanation("DrawdownGuard", state)

        assert "drawdown" in explanation

    def test_volatility_guard(self) -> None:
        """Test volatility guard explanation."""
        state = RiskState()
        explanation = get_guard_explanation("VolatilityGuard", state)

        assert "volatility" in explanation

    def test_rate_limit_guard(self) -> None:
        """Test rate limit guard explanation."""
        state = RiskState()
        explanation = get_guard_explanation("RateLimitGuard", state)

        assert "rate limit" in explanation


class TestUnknownGuardFallback:
    """Tests for unknown guard fallback behavior."""

    def test_unknown_guard_returns_fallback(self) -> None:
        """Test unknown guard names return generic fallback."""
        state = RiskState()
        explanation = get_guard_explanation("SomeUnknownGuard", state)

        assert "Guard active" in explanation
        assert "not available" in explanation

    def test_empty_guard_name_returns_fallback(self) -> None:
        """Test empty guard name returns fallback."""
        state = RiskState()
        explanation = get_guard_explanation("", state)

        assert "Guard active" in explanation


class TestGetGuardExplanations:
    """Tests for batch guard explanation function."""

    def test_multiple_guards(self) -> None:
        """Test getting explanations for multiple guards."""
        state = RiskState(
            daily_loss_limit_pct=0.05,
            current_daily_loss_pct=-0.02,
            max_leverage=3.0,
        )
        guards = ["DailyLossGuard", "MaxLeverageGuard"]

        explanations = get_guard_explanations(guards, state)

        assert len(explanations) == 2
        assert explanations[0][0] == "DailyLossGuard"
        assert "75%" in explanations[0][1]
        assert explanations[1][0] == "MaxLeverageGuard"
        assert "3.0x" in explanations[1][1]

    def test_empty_guard_list(self) -> None:
        """Test getting explanations for empty guard list."""
        state = RiskState()
        explanations = get_guard_explanations([], state)

        assert explanations == []

    def test_mixed_known_and_unknown(self) -> None:
        """Test getting explanations for mix of known and unknown guards."""
        state = RiskState(max_leverage=2.0)
        guards = ["LeverageGuard", "CustomGuard"]

        explanations = get_guard_explanations(guards, state)

        assert len(explanations) == 2
        assert "2.0x" in explanations[0][1]
        assert "not available" in explanations[1][1]

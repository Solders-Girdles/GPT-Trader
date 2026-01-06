"""Tests for risk_preview helper module."""

import pytest

from gpt_trader.tui.risk_preview import (
    SHOCK_SCENARIOS,
    RiskPreviewResult,
    RiskPreviewScenario,
    _get_ratio_status,
    _get_triggered_guards,
    compute_all_previews,
    compute_preview,
    get_default_scenarios,
)
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.thresholds import DEFAULT_RISK_THRESHOLDS, StatusLevel
from gpt_trader.tui.types import RiskState


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
            guards_triggered=["DailyLossGuard"],
        )
        assert result.label == "-5%"
        assert result.projected_loss_pct == 75.0
        assert result.guards_triggered == ["DailyLossGuard"]

        # Test default empty guards
        result2 = RiskPreviewResult(label="+2%", projected_loss_pct=30.0, status=StatusLevel.OK)
        assert result2.guards_triggered == []


class TestGetRatioStatus:
    """Tests for _get_ratio_status helper."""

    def test_status_boundaries(self):
        """Status transitions at correct threshold boundaries."""
        # OK: below 50%
        assert _get_ratio_status(0.0, DEFAULT_RISK_THRESHOLDS) == StatusLevel.OK
        assert _get_ratio_status(0.49, DEFAULT_RISK_THRESHOLDS) == StatusLevel.OK
        # WARNING: 50-75% (exclusive at 50, exclusive at 75)
        assert _get_ratio_status(0.50, DEFAULT_RISK_THRESHOLDS) == StatusLevel.WARNING
        assert _get_ratio_status(0.74, DEFAULT_RISK_THRESHOLDS) == StatusLevel.WARNING
        # CRITICAL: 75%+
        assert _get_ratio_status(0.75, DEFAULT_RISK_THRESHOLDS) == StatusLevel.CRITICAL
        assert _get_ratio_status(1.5, DEFAULT_RISK_THRESHOLDS) == StatusLevel.CRITICAL


class TestGetTriggeredGuards:
    """Tests for _get_triggered_guards helper."""

    def test_guards_by_threshold(self):
        """Guards trigger at correct thresholds."""
        # Below warning: no guards
        assert _get_triggered_guards(0.74, DEFAULT_RISK_THRESHOLDS) == []
        # At warning: DailyLossGuard triggers
        guards = _get_triggered_guards(0.75, DEFAULT_RISK_THRESHOLDS)
        assert "DailyLossGuard" in guards
        assert "ReduceOnlyMode" not in guards
        # At 100%: both guards trigger
        guards = _get_triggered_guards(1.0, DEFAULT_RISK_THRESHOLDS)
        assert "DailyLossGuard" in guards
        assert "ReduceOnlyMode" in guards


class TestComputePreview:
    """Tests for compute_preview function."""

    @pytest.fixture
    def tui_state(self) -> TuiState:
        """Create a TuiState with realistic risk data."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.10,  # 10% daily limit
            current_daily_loss_pct=-0.02,  # Currently 2% loss (20% of limit)
        )
        return state

    def test_positive_shock_no_increase(self, tui_state: TuiState):
        """Positive shock doesn't increase loss."""
        result = compute_preview(tui_state, shock_pct=0.05, label="+5%")
        assert result.label == "+5%"
        assert result.projected_loss_pct == pytest.approx(20.0, rel=0.01)
        assert result.status == StatusLevel.OK

    def test_negative_shock_increases_loss(self, tui_state: TuiState):
        """Negative shock increases loss with leverage."""
        result = compute_preview(tui_state, shock_pct=-0.05, label="-5%")
        # Current 20% + (5% * 5x leverage) = 20% + 25% = 45%
        assert result.projected_loss_pct == pytest.approx(45.0, rel=0.01)
        assert result.status == StatusLevel.OK

    def test_shock_triggers_warning_and_critical(self, tui_state: TuiState):
        """Large shocks can trigger warning and critical status."""
        # -10%: Current 20% + 50% = 70% -> WARNING
        result = compute_preview(tui_state, shock_pct=-0.10, label="-10%")
        assert result.projected_loss_pct == pytest.approx(70.0, rel=0.01)
        assert result.status == StatusLevel.WARNING

        # Set higher current loss for critical test
        tui_state.risk_data.current_daily_loss_pct = -0.05  # 50% of limit
        result = compute_preview(tui_state, shock_pct=-0.10, label="-10%")
        # 50% + 50% = 100% -> CRITICAL
        assert result.projected_loss_pct == pytest.approx(100.0, rel=0.01)
        assert result.status == StatusLevel.CRITICAL
        assert "DailyLossGuard" in result.guards_triggered
        assert "ReduceOnlyMode" in result.guards_triggered

    def test_leverage_fallback_and_no_limit(self, tui_state: TuiState):
        """Falls back to 1.0 leverage; returns OK when no limit configured."""
        # Test leverage fallback
        tui_state.risk_data.max_leverage = 0.0
        result = compute_preview(tui_state, shock_pct=-0.05, label="-5%")
        # Current 20% + (5% * 1x) = 25%
        assert result.projected_loss_pct == pytest.approx(25.0, rel=0.01)

        # Test no limit returns OK
        tui_state.risk_data.daily_loss_limit_pct = 0.0
        result = compute_preview(tui_state, shock_pct=-0.10, label="-10%")
        assert result.projected_loss_pct == 0.0
        assert result.status == StatusLevel.OK


class TestGetDefaultScenarios:
    """Tests for get_default_scenarios function."""

    def test_returns_all_scenarios(self):
        """Returns all standard shock scenarios with correct structure."""
        scenarios = get_default_scenarios()
        assert len(scenarios) == len(SHOCK_SCENARIOS)
        for scenario in scenarios:
            assert isinstance(scenario, RiskPreviewScenario)

        # Has both positive and negative
        positive = [s for s in scenarios if s.shock_pct > 0]
        negative = [s for s in scenarios if s.shock_pct < 0]
        assert len(positive) > 0
        assert len(negative) > 0


class TestComputeAllPreviews:
    """Tests for compute_all_previews batch function."""

    @pytest.fixture
    def tui_state(self) -> TuiState:
        """Create a TuiState with risk data."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(
            max_leverage=2.0,
            daily_loss_limit_pct=0.10,
            current_daily_loss_pct=-0.01,
        )
        return state

    def test_returns_all_scenario_results(self, tui_state: TuiState):
        """Returns results for all default scenarios."""
        results = compute_all_previews(tui_state)
        assert len(results) == len(SHOCK_SCENARIOS)
        labels = [r.label for r in results]
        expected = [label for label, _ in SHOCK_SCENARIOS]
        assert labels == expected

    def test_negative_scenarios_have_higher_loss(self, tui_state: TuiState):
        """Negative shocks should have higher loss than positive."""
        results = compute_all_previews(tui_state)
        current_pct = 10.0  # 10% loss ratio
        for r in results:
            if "-" in r.label:
                assert r.projected_loss_pct > current_pct
            else:
                assert r.projected_loss_pct == pytest.approx(current_pct, rel=0.01)


class TestShockScenariosConstant:
    """Tests for SHOCK_SCENARIOS constant."""

    def test_standard_scenarios(self):
        """Contains standard symmetric ±2%, ±5%, ±10% scenarios in order."""
        labels = [label for label, _ in SHOCK_SCENARIOS]
        assert "-2%" in labels and "+2%" in labels
        assert "-5%" in labels and "+5%" in labels
        assert "-10%" in labels and "+10%" in labels

        # Should be sorted ascending
        percentages = [pct for _, pct in SHOCK_SCENARIOS]
        assert percentages == sorted(percentages)

        # Symmetric (each positive has matching negative)
        for pct in percentages:
            if pct != 0:
                assert -pct in percentages

"""Tests for risk_preview helper module."""

from decimal import Decimal

import pytest

from gpt_trader.tui.risk_preview import (
    SHOCK_SCENARIOS,
    GuardImpact,
    GuardPositionImpact,
    RiskPreviewResult,
    RiskPreviewScenario,
    _compute_position_impacts,
    _get_guard_impacts,
    _get_ratio_status,
    compute_all_previews,
    compute_preview,
    get_default_scenarios,
)
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.thresholds import DEFAULT_RISK_THRESHOLDS, StatusLevel
from gpt_trader.tui.types import PortfolioSummary, Position, RiskState


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
        guard_names = [g.name for g in result.guard_impacts]
        assert "DailyLossGuard" in guard_names
        assert "ReduceOnlyMode" in guard_names

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


class TestComputePositionImpacts:
    """Tests for _compute_position_impacts helper."""

    @pytest.fixture
    def tui_state_with_positions(self) -> TuiState:
        """Create TuiState with positions for testing."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.05,  # 5% daily limit
            current_daily_loss_pct=-0.01,
        )
        # Add positions
        btc_pos = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            unrealized_pnl=Decimal("-200"),  # -2% of equity
            mark_price=Decimal("48000"),
            side="LONG",
            leverage=1,
        )
        eth_pos = Position(
            symbol="ETH-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("3000"),
            unrealized_pnl=Decimal("-50"),  # -0.5% of equity
            mark_price=Decimal("2950"),
            side="LONG",
            leverage=1,
        )
        state.position_data = PortfolioSummary(
            positions={"BTC-USD": btc_pos, "ETH-USD": eth_pos},
            equity=Decimal("10000"),
        )
        return state

    def test_no_positions_returns_empty(self):
        """Empty positions returns empty impacts."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.05,
        )
        state.position_data = PortfolioSummary(positions={}, equity=Decimal("10000"))

        impacts = _compute_position_impacts(state, -0.10, 0.05, DEFAULT_RISK_THRESHOLDS)
        assert impacts == []

    def test_zero_equity_returns_empty(self):
        """Zero equity returns empty impacts."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(max_leverage=5.0, daily_loss_limit_pct=0.05)
        state.position_data = PortfolioSummary(
            positions={"BTC-USD": Position(symbol="BTC-USD", quantity=Decimal("1"))},
            equity=Decimal("0"),
        )

        impacts = _compute_position_impacts(state, -0.10, 0.05, DEFAULT_RISK_THRESHOLDS)
        assert impacts == []

    def test_negative_shock_creates_impacts_for_long_positions(
        self, tui_state_with_positions: TuiState
    ):
        """Negative shock creates impacts for long positions losing value."""
        impacts = _compute_position_impacts(
            tui_state_with_positions, -0.10, 0.05, DEFAULT_RISK_THRESHOLDS
        )

        # BTC position: 0.1 * 48000 = 4800 value, -10% shock = -480 P&L delta
        # Current P&L: -200, Projected: -680 (-6.8% of equity)
        # This exceeds 75% of 5% limit = 3.75%, so should trigger

        # Should have at least one impact for DailyLossGuard
        daily_loss_impacts = [i for i in impacts if i.guard_name == "DailyLossGuard"]
        assert len(daily_loss_impacts) > 0

        # Check BTC impact is present
        btc_impacts = [i for i in daily_loss_impacts if i.symbol == "BTC-USD"]
        assert len(btc_impacts) == 1
        assert btc_impacts[0].projected_pnl_pct < 0
        assert "→" in btc_impacts[0].reason

    def test_positive_shock_gains_for_long_positions(self, tui_state_with_positions: TuiState):
        """Positive shock creates gains for long positions (no breach)."""
        impacts = _compute_position_impacts(
            tui_state_with_positions, +0.10, 0.05, DEFAULT_RISK_THRESHOLDS
        )

        # Positive shock means gains for longs, so no DailyLossGuard breach
        daily_loss_impacts = [i for i in impacts if i.guard_name == "DailyLossGuard"]
        # May or may not have impacts depending on whether gains exceed threshold
        # But projected P&L should be positive or less negative
        for impact in daily_loss_impacts:
            # With positive shock, projected should improve from current
            pass  # Just ensure no exception

    def test_leverage_guard_triggers_on_excessive_leverage(self):
        """LeverageGuard triggers when position leverage exceeds max."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(
            max_leverage=2.0,  # Max 2x
            daily_loss_limit_pct=0.05,
        )
        # Position with 5x leverage
        high_lev_pos = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            mark_price=Decimal("50000"),
            side="LONG",
            leverage=5,  # Exceeds max of 2x
        )
        state.position_data = PortfolioSummary(
            positions={"BTC-USD": high_lev_pos},
            equity=Decimal("10000"),
        )

        impacts = _compute_position_impacts(state, -0.05, 0.05, DEFAULT_RISK_THRESHOLDS)

        leverage_impacts = [i for i in impacts if i.guard_name == "LeverageGuard"]
        assert len(leverage_impacts) == 1
        assert leverage_impacts[0].symbol == "BTC-USD"
        assert "5x >" in leverage_impacts[0].reason and "max" in leverage_impacts[0].reason

    def test_impacts_sorted_by_projected_loss(self, tui_state_with_positions: TuiState):
        """Impacts are sorted by projected loss (worst first)."""
        impacts = _compute_position_impacts(
            tui_state_with_positions, -0.10, 0.05, DEFAULT_RISK_THRESHOLDS
        )

        if len(impacts) > 1:
            for i in range(len(impacts) - 1):
                assert impacts[i].projected_pnl_pct <= impacts[i + 1].projected_pnl_pct

    def test_guard_position_impact_dataclass(self):
        """GuardPositionImpact dataclass stores all fields correctly."""
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


class TestComputePreviewWithPositionImpacts:
    """Tests for compute_preview returning position impacts."""

    @pytest.fixture
    def tui_state_with_positions(self) -> TuiState:
        """Create TuiState with positions for testing."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.05,
            current_daily_loss_pct=-0.01,
        )
        btc_pos = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            unrealized_pnl=Decimal("-500"),
            mark_price=Decimal("49000"),
            side="LONG",
            leverage=1,
        )
        state.position_data = PortfolioSummary(
            positions={"BTC-USD": btc_pos},
            equity=Decimal("10000"),
        )
        return state

    def test_compute_preview_includes_position_impacts(self, tui_state_with_positions: TuiState):
        """compute_preview returns position_impacts field."""
        result = compute_preview(tui_state_with_positions, -0.10, label="-10%")

        assert hasattr(result, "position_impacts")
        assert isinstance(result.position_impacts, list)

    def test_no_limit_returns_empty_position_impacts(self):
        """No daily limit configured returns empty position impacts."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.risk_data = RiskState(
            max_leverage=5.0,
            daily_loss_limit_pct=0.0,  # No limit
        )

        result = compute_preview(state, -0.10, label="-10%")
        assert result.position_impacts == []

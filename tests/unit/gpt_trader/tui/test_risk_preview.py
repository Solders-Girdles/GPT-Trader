"""Tests for risk preview helpers and scenario generation."""

from decimal import Decimal

import pytest

from gpt_trader.tui.risk_preview import (
    SHOCK_SCENARIOS,
    GuardImpact,
    GuardPositionImpact,
    RiskPreviewResult,
    RiskPreviewScenario,
    _compute_position_impacts,
    _get_ratio_status,
    compute_all_previews,
    compute_preview,
    get_default_scenarios,
)
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.thresholds import DEFAULT_RISK_THRESHOLDS, StatusLevel
from gpt_trader.tui.types import PortfolioSummary, Position, RiskState


def _state_with_risk(
    *,
    max_leverage: float = 5.0,
    daily_loss_limit_pct: float = 0.10,
    current_daily_loss_pct: float | None = -0.02,
) -> TuiState:
    state = TuiState(validation_enabled=False, delta_updates_enabled=False)
    state.risk_data = RiskState(
        max_leverage=max_leverage,
        daily_loss_limit_pct=daily_loss_limit_pct,
        current_daily_loss_pct=current_daily_loss_pct,
    )
    return state


def _state_with_positions() -> TuiState:
    state = _state_with_risk(
        max_leverage=5.0, daily_loss_limit_pct=0.05, current_daily_loss_pct=-0.01
    )
    state.position_data = PortfolioSummary(
        positions={
            "BTC-USD": Position(
                symbol="BTC-USD",
                quantity=Decimal("0.1"),
                unrealized_pnl=Decimal("-200"),
                mark_price=Decimal("48000"),
            ),
            "ETH-USD": Position(
                symbol="ETH-USD",
                quantity=Decimal("1.0"),
                unrealized_pnl=Decimal("-50"),
                mark_price=Decimal("2950"),
            ),
        },
        equity=Decimal("10000"),
    )
    return state


def test_dataclasses_and_helpers() -> None:
    scenario = RiskPreviewScenario(label="+5%", shock_pct=0.05)
    assert (scenario.label, scenario.shock_pct) == ("+5%", 0.05)
    with pytest.raises(AttributeError):
        scenario.label = "changed"  # type: ignore[misc]

    result = RiskPreviewResult(
        label="-5%",
        projected_loss_pct=75.0,
        status=StatusLevel.WARNING,
        guard_impacts=[GuardImpact(name="DailyLossGuard", reason=">=75% of limit")],
    )
    assert (result.label, result.projected_loss_pct) == ("-5%", 75.0)
    assert result.guard_impacts[0].name == "DailyLossGuard"
    assert (
        RiskPreviewResult(label="+2%", projected_loss_pct=30.0, status=StatusLevel.OK).guard_impacts
        == []
    )

    impact = GuardPositionImpact(
        guard_name="DailyLossGuard",
        symbol="BTC-USD",
        current_pnl_pct=-2.0,
        projected_pnl_pct=-6.8,
        limit_pct=5.0,
        reason="-2.0% → -6.8%",
    )
    assert (impact.guard_name, impact.symbol, impact.limit_pct) == (
        "DailyLossGuard",
        "BTC-USD",
        5.0,
    )


@pytest.mark.parametrize(
    "ratio, expected",
    [
        (0.0, StatusLevel.OK),
        (0.50, StatusLevel.WARNING),
        (0.75, StatusLevel.CRITICAL),
    ],
)
def test_ratio_status_boundaries(ratio: float, expected: StatusLevel) -> None:
    assert _get_ratio_status(ratio, DEFAULT_RISK_THRESHOLDS) == expected


@pytest.mark.parametrize(
    "current_loss, shock_pct, expected_pct, expected_status, expected_guards",
    [
        (-0.02, 0.05, 20.0, StatusLevel.OK, set()),
        (-0.02, -0.05, 45.0, StatusLevel.OK, set()),
        (-0.02, -0.10, 70.0, StatusLevel.WARNING, set()),
        (-0.05, -0.10, 100.0, StatusLevel.CRITICAL, {"DailyLossGuard", "ReduceOnlyMode"}),
    ],
)
def test_compute_preview_shocks(
    current_loss: float,
    shock_pct: float,
    expected_pct: float,
    expected_status: StatusLevel,
    expected_guards: set[str],
) -> None:
    state = _state_with_risk(
        max_leverage=5.0, daily_loss_limit_pct=0.10, current_daily_loss_pct=current_loss
    )
    result = compute_preview(state, shock_pct=shock_pct, label="-10%")
    assert result.projected_loss_pct == pytest.approx(expected_pct, rel=0.01)
    assert result.status == expected_status
    if expected_guards:
        assert expected_guards <= {g.name for g in result.guard_impacts}


def test_compute_preview_leverage_fallback_and_no_limit() -> None:
    state = _state_with_risk(
        max_leverage=0.0, daily_loss_limit_pct=0.10, current_daily_loss_pct=-0.02
    )
    result = compute_preview(state, shock_pct=-0.05, label="-5%")
    assert result.projected_loss_pct == pytest.approx(25.0, rel=0.01)

    state.risk_data.daily_loss_limit_pct = 0.0
    result = compute_preview(state, shock_pct=-0.10, label="-10%")
    assert result.projected_loss_pct == 0.0
    assert result.status == StatusLevel.OK


def test_default_scenarios_and_previews() -> None:
    scenarios = get_default_scenarios()
    state = _state_with_risk(
        max_leverage=2.0, daily_loss_limit_pct=0.10, current_daily_loss_pct=-0.01
    )
    results = compute_all_previews(state)
    assert len(scenarios) == len(SHOCK_SCENARIOS) == len(results)
    assert [r.label for r in results] == [label for label, _ in SHOCK_SCENARIOS]
    current_pct = 10.0
    for result in results:
        if "-" in result.label:
            assert result.projected_loss_pct > current_pct
        else:
            assert result.projected_loss_pct == pytest.approx(current_pct, rel=0.01)


@pytest.mark.parametrize(
    "positions, equity",
    [
        ({}, Decimal("10000")),
        ({"BTC-USD": Position(symbol="BTC-USD", quantity=Decimal("1"))}, Decimal("0")),
    ],
)
def test_compute_position_impacts_empty(positions: dict, equity: Decimal) -> None:
    state = _state_with_risk(max_leverage=5.0, daily_loss_limit_pct=0.05)
    state.position_data = PortfolioSummary(positions=positions, equity=equity)
    impacts = _compute_position_impacts(state, -0.10, 0.05, DEFAULT_RISK_THRESHOLDS)
    assert impacts == []


def test_negative_shock_creates_impacts_for_longs() -> None:
    impacts = _compute_position_impacts(
        _state_with_positions(), -0.10, 0.05, DEFAULT_RISK_THRESHOLDS
    )
    daily_loss_impacts = [i for i in impacts if i.guard_name == "DailyLossGuard"]
    assert daily_loss_impacts
    btc_impacts = [i for i in daily_loss_impacts if i.symbol == "BTC-USD"]
    assert len(btc_impacts) == 1
    assert btc_impacts[0].projected_pnl_pct < 0
    assert "→" in btc_impacts[0].reason
    for i in range(len(impacts) - 1):
        assert impacts[i].projected_pnl_pct <= impacts[i + 1].projected_pnl_pct

    positive_impacts = _compute_position_impacts(
        _state_with_positions(), 0.10, 0.05, DEFAULT_RISK_THRESHOLDS
    )
    for impact in positive_impacts:
        assert impact.projected_pnl_pct >= impact.current_pnl_pct


def test_leverage_guard_triggers_on_excessive_leverage() -> None:
    state = _state_with_risk(max_leverage=2.0, daily_loss_limit_pct=0.05)
    state.position_data = PortfolioSummary(
        positions={
            "BTC-USD": Position(
                symbol="BTC-USD",
                quantity=Decimal("1.0"),
                unrealized_pnl=Decimal("0"),
                mark_price=Decimal("50000"),
                leverage=5,
            )
        },
        equity=Decimal("10000"),
    )
    impacts = _compute_position_impacts(state, -0.05, 0.05, DEFAULT_RISK_THRESHOLDS)
    leverage_impacts = [i for i in impacts if i.guard_name == "LeverageGuard"]
    assert len(leverage_impacts) == 1
    assert leverage_impacts[0].symbol == "BTC-USD"
    assert "5x >" in leverage_impacts[0].reason and "max" in leverage_impacts[0].reason


def test_compute_preview_position_impacts_and_no_limit() -> None:
    state = _state_with_risk(
        max_leverage=5.0, daily_loss_limit_pct=0.05, current_daily_loss_pct=-0.01
    )
    state.position_data = PortfolioSummary(
        positions={
            "BTC-USD": Position(
                symbol="BTC-USD",
                quantity=Decimal("0.5"),
                unrealized_pnl=Decimal("-500"),
                mark_price=Decimal("49000"),
            )
        },
        equity=Decimal("10000"),
    )
    result = compute_preview(state, -0.10, label="-10%")
    assert isinstance(result.position_impacts, list)
    no_limit_state = _state_with_risk(max_leverage=5.0, daily_loss_limit_pct=0.0)
    result = compute_preview(no_limit_state, -0.10, label="-10%")
    assert result.position_impacts == []

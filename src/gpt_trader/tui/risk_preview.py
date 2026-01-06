"""Risk preview calculations for UI-only scenario analysis.

Provides helper functions and dataclasses for displaying projected
risk metrics under various shock scenarios in the RiskDetailModal.

This module is UI-only and does not modify any core risk logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gpt_trader.tui.thresholds import (
    DEFAULT_RISK_THRESHOLDS,
    RiskThresholds,
    StatusLevel,
)

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState


# Standard shock scenarios for preview chips
SHOCK_SCENARIOS: list[tuple[str, float]] = [
    ("-10%", -0.10),
    ("-5%", -0.05),
    ("-2%", -0.02),
    ("+2%", +0.02),
    ("+5%", +0.05),
    ("+10%", +0.10),
]


@dataclass(frozen=True)
class RiskPreviewScenario:
    """A scenario configuration for risk preview.

    Attributes:
        label: Human-readable label (e.g., "+5%", "-2%").
        shock_pct: Shock percentage as decimal (e.g., +0.05, -0.02).
    """

    label: str
    shock_pct: float


@dataclass(frozen=True)
class GuardImpact:
    """A guard that would trigger with its reason.

    Attributes:
        name: Guard name (e.g., "DailyLossGuard").
        reason: Human-readable reason (e.g., ">=75% of limit").
    """

    name: str
    reason: str


@dataclass
class RiskPreviewResult:
    """Result of computing a risk preview scenario.

    Attributes:
        label: Scenario label for display.
        projected_loss_pct: Projected loss as percentage of limit (0-100+).
        status: Status level (OK/WARNING/CRITICAL) based on thresholds.
        guard_impacts: List of guards that would trigger with reasons.
    """

    label: str
    projected_loss_pct: float
    status: StatusLevel
    guard_impacts: list[GuardImpact] = field(default_factory=list)


def compute_preview(
    state: TuiState,
    shock_pct: float,
    label: str = "",
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
) -> RiskPreviewResult:
    """Compute risk preview for a given shock scenario.

    This is a UI-only calculation that projects what the risk metrics
    would look like if the market moved by the given shock percentage.

    The projection is conservative:
    - Projected loss ratio = current + abs(shock_pct * leverage_multiplier)
    - Uses max_leverage as multiplier (fallback to 1.0 if not set)
    - Negative shocks (drawdowns) increase loss ratio
    - Positive shocks are ignored for loss calculation (we care about drawdown risk)

    Args:
        state: Current TUI state containing risk data.
        shock_pct: Shock percentage as decimal (e.g., -0.05 for -5%).
        label: Optional label for the scenario.
        thresholds: Risk thresholds for status calculation.

    Returns:
        RiskPreviewResult with projected loss ratio and status.
    """
    risk_data = state.risk_data

    # Current loss as ratio of limit (0-1)
    current_loss_pct = abs(risk_data.current_daily_loss_pct)
    limit_pct = risk_data.daily_loss_limit_pct

    if limit_pct <= 0:
        # No limit configured, can't compute meaningful projection
        return RiskPreviewResult(
            label=label,
            projected_loss_pct=0.0,
            status=StatusLevel.OK,
            guard_impacts=[],
        )

    current_ratio = current_loss_pct / limit_pct

    # Leverage multiplier (max_leverage or 1.0)
    leverage = risk_data.max_leverage if risk_data.max_leverage > 0 else 1.0

    # For loss projection, we only care about negative shocks (drawdowns)
    # A positive move doesn't increase our loss
    if shock_pct >= 0:
        # Positive shock: no additional loss projected
        projected_ratio = current_ratio
    else:
        # Negative shock: project additional loss impact
        # shock_pct is negative, so abs() makes it positive
        shock_impact = abs(shock_pct) * leverage
        projected_ratio = current_ratio + shock_impact

    # Convert to percentage for display (0-100 scale)
    projected_pct = projected_ratio * 100

    # Determine status using thresholds
    status = _get_ratio_status(projected_ratio, thresholds)

    # Determine which guards would trigger with reasons
    guard_impacts = _get_guard_impacts(projected_ratio, thresholds)

    return RiskPreviewResult(
        label=label,
        projected_loss_pct=projected_pct,
        status=status,
        guard_impacts=guard_impacts,
    )


def _get_ratio_status(loss_ratio: float, thresholds: RiskThresholds) -> StatusLevel:
    """Get status level for a loss ratio.

    Args:
        loss_ratio: Loss as ratio of limit (0-1, can exceed 1).
        thresholds: Thresholds for status boundaries.

    Returns:
        StatusLevel based on loss ratio.
    """
    if loss_ratio < thresholds.loss_ratio_ok:
        return StatusLevel.OK
    elif loss_ratio < thresholds.loss_ratio_warn:
        return StatusLevel.WARNING
    return StatusLevel.CRITICAL


def _get_guard_impacts(loss_ratio: float, thresholds: RiskThresholds) -> list[GuardImpact]:
    """Determine which guards would trigger at the given loss ratio with reasons.

    This is a simplified UI projection - the actual guards have more
    complex logic, but this gives users a sense of what might trip.

    Args:
        loss_ratio: Projected loss as ratio of limit (0-1, can exceed 1).
        thresholds: Thresholds for guard boundaries.

    Returns:
        List of GuardImpact objects with names and reasons.
    """
    impacts = []

    # DailyLossGuard triggers at warning threshold (75% by default)
    if loss_ratio >= thresholds.loss_ratio_warn:
        warn_pct = int(thresholds.loss_ratio_warn * 100)
        impacts.append(GuardImpact(name="DailyLossGuard", reason=f">={warn_pct}% of limit"))

    # At critical (100%+), reduce-only would likely engage
    if loss_ratio >= 1.0:
        impacts.append(GuardImpact(name="ReduceOnlyMode", reason=">=100% of limit"))

    return impacts


def get_default_scenarios() -> list[RiskPreviewScenario]:
    """Get the default shock scenarios for preview chips.

    Returns:
        List of RiskPreviewScenario for standard chips.
    """
    return [RiskPreviewScenario(label=label, shock_pct=pct) for label, pct in SHOCK_SCENARIOS]


def compute_all_previews(
    state: TuiState,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
) -> list[RiskPreviewResult]:
    """Compute previews for all default shock scenarios.

    Convenience function for computing all standard scenarios at once.

    Args:
        state: Current TUI state containing risk data.
        thresholds: Risk thresholds for status calculation.

    Returns:
        List of RiskPreviewResult for all default scenarios.
    """
    results = []
    for label, shock_pct in SHOCK_SCENARIOS:
        result = compute_preview(state, shock_pct, label=label, thresholds=thresholds)
        results.append(result)
    return results

"""Position sizing delegation."""

from __future__ import annotations

from collections.abc import Callable

from bot_v2.features.live_trade.risk.position_sizing import (
    ImpactAssessment,
    ImpactRequest,
    PositionSizingAdvice,
    PositionSizingContext,
)


class LiveRiskManagerSizingMixin:
    """Delegate position sizing to the dedicated service."""

    def size_position(self, context: PositionSizingContext) -> PositionSizingAdvice:
        """Calculate position size using dynamic estimator or fallback logic."""
        return self.position_sizer.size_position(context)

    def set_impact_estimator(
        self, estimator: Callable[[ImpactRequest], ImpactAssessment] | None
    ) -> None:
        """Install or clear the market-impact estimator hook."""
        self.position_sizer.set_impact_estimator(estimator)
        self.pre_trade_validator._impact_estimator = estimator
        self._impact_estimator = estimator


__all__ = ["LiveRiskManagerSizingMixin"]

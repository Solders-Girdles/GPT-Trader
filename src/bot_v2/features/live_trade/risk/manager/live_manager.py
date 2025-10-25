"""Live trading risk manager facade assembled from focused mixins."""

from __future__ import annotations

from bot_v2.features.live_trade.risk.position_sizing import (
    ImpactAssessment,
    ImpactRequest,
    PositionSizingAdvice,
    PositionSizingContext,
)
from bot_v2.features.live_trade.risk.pre_trade_checks import (
    PreTradeValidator,
    ValidationError,
)
from bot_v2.features.live_trade.risk.state_management import (
    RiskRuntimeState,
    StateManager,
)

from .events import LiveRiskManagerEventMixin
from .initialization import LiveRiskManagerInitializationMixin
from .positions import LiveRiskManagerPositionMixin
from .runtime import LiveRiskManagerRuntimeMixin
from .sizing import LiveRiskManagerSizingMixin
from .state import LiveRiskManagerStateMixin
from .validation import LiveRiskManagerValidationMixin


class LiveRiskManager(
    LiveRiskManagerValidationMixin,
    LiveRiskManagerSizingMixin,
    LiveRiskManagerRuntimeMixin,
    LiveRiskManagerPositionMixin,
    LiveRiskManagerStateMixin,
    LiveRiskManagerEventMixin,
    LiveRiskManagerInitializationMixin,
):
    """Risk management for perpetuals live trading."""


__all__ = [
    "LiveRiskManager",
    "ValidationError",
    "PositionSizingContext",
    "PositionSizingAdvice",
    "ImpactRequest",
    "ImpactAssessment",
    "RiskRuntimeState",
    "StateManager",
    "PreTradeValidator",
]

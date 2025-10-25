"""Risk manager package exposing the live trading facade."""

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
from bot_v2.orchestration.configuration import RiskConfig

from .circuit_breaker import CircuitBreakerStateAdapter
from .live_manager import LiveRiskManager
from .logging import logger
from .registries import MarkTimestampRegistry

__all__ = [
    "LiveRiskManager",
    "MarkTimestampRegistry",
    "CircuitBreakerStateAdapter",
    "ValidationError",
    "PositionSizingContext",
    "PositionSizingAdvice",
    "ImpactRequest",
    "ImpactAssessment",
    "RiskRuntimeState",
    "StateManager",
    "PreTradeValidator",
    "RiskConfig",
    "logger",
]

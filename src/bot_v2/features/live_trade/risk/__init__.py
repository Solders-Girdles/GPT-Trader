"""
Risk management subpackage.

Exports all public classes and dataclasses for backward compatibility.
"""

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk.manager import LiveRiskManager
from bot_v2.features.live_trade.risk.position_sizing import (
    ImpactAssessment,
    ImpactRequest,
    PositionSizer,
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
from bot_v2.features.live_trade.risk_runtime import RuntimeMonitor

__all__ = [
    # Main facade
    "LiveRiskManager",
    # Configuration (re-exported for backward compatibility)
    "RiskConfig",
    # Position Sizing
    "PositionSizer",
    "PositionSizingContext",
    "PositionSizingAdvice",
    "ImpactRequest",
    "ImpactAssessment",
    # Pre-Trade Validation
    "PreTradeValidator",
    "ValidationError",
    # Runtime Monitoring
    "RuntimeMonitor",
    # State Management
    "StateManager",
    "RiskRuntimeState",
]

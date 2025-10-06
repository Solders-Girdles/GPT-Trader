"""
Risk management subpackage.

Exports all public classes and dataclasses for backward compatibility.
"""

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk.gate_validator import RiskGateValidator
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
from bot_v2.features.live_trade.risk.runtime_monitoring import RuntimeMonitor
from bot_v2.features.live_trade.risk.state_management import (
    RiskRuntimeState,
    RiskStateManager,
)

# Backward compatibility alias
StateManager = RiskStateManager

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
    # Gate Validation
    "RiskGateValidator",
    # State Management
    "RiskStateManager",
    "StateManager",  # Backward compatibility alias
    "RiskRuntimeState",
]

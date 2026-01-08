"""
Backward compatibility re-export for RiskConfig.

The canonical location is now:
    gpt_trader.features.live_trade.risk.config

This module re-exports RiskConfig for backward compatibility with existing
imports from orchestration.configuration.risk.model.
"""

# Re-export from canonical location
from gpt_trader.features.live_trade.risk.config import (
    RISK_CONFIG_ENV_ALIASES,
    RISK_CONFIG_ENV_KEYS,
    RiskConfig,
)

__all__ = ["RiskConfig", "RISK_CONFIG_ENV_KEYS", "RISK_CONFIG_ENV_ALIASES"]

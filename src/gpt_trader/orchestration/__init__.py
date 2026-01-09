"""
DEPRECATED: The orchestration package is deprecated.

All functionality has moved to canonical locations:
- BotConfig → gpt_trader.app.config
- bootstrap → gpt_trader.app.bootstrap
- SpotProfileService → gpt_trader.features.live_trade.orchestrator.spot_profile_service
- IntxPortfolioService → gpt_trader.features.brokerages.coinbase.intx_portfolio_service

This package only re-exports for backwards compatibility.
"""

import warnings

from gpt_trader.orchestration.configuration import BotConfig

__all__ = ["BotConfig"]

# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration is deprecated. "
    "Import from gpt_trader.app or gpt_trader.features instead.",
    DeprecationWarning,
    stacklevel=2,
)

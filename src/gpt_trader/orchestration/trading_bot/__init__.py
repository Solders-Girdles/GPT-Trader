"""
DEPRECATED: Trading bot package has moved to gpt_trader.features.live_trade.bot

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.live_trade.bot import TradingBot
"""

import warnings

from gpt_trader.features.live_trade.bot import TradingBot

warnings.warn(
    "gpt_trader.orchestration.trading_bot is deprecated. "
    "Import from gpt_trader.features.live_trade.bot instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TradingBot"]

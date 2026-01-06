"""
DEPRECATED: BotConfig has moved to gpt_trader.app.config.

This module re-exports from the new location for backwards compatibility.
Update your imports to use:

    from gpt_trader.app.config import BotConfig, BotRiskConfig

This shim will be removed in a future release.
"""

import warnings

# Re-export everything from the new canonical location
from gpt_trader.app.config.bot_config import (
    DEFAULT_SPOT_RISK_PATH,
    DEFAULT_SPOT_SYMBOLS,
    TOP_VOLUME_BASES,
    BotConfig,
    BotRiskConfig,
    MeanReversionConfig,
    StrategyType,
)

__all__ = [
    "BotConfig",
    "BotRiskConfig",
    "MeanReversionConfig",
    "StrategyType",
    "DEFAULT_SPOT_RISK_PATH",
    "DEFAULT_SPOT_SYMBOLS",
    "TOP_VOLUME_BASES",
]


def __getattr__(name: str):
    """Emit deprecation warning on first access."""
    if name in __all__:
        warnings.warn(
            f"Importing {name} from gpt_trader.orchestration.configuration.bot_config "
            f"is deprecated. Use gpt_trader.app.config instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

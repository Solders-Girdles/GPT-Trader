"""
DEPRECATED: BotConfig has moved to gpt_trader.app.config.

This module re-exports from the new location for backwards compatibility.
Update your imports to use:

    from gpt_trader.app.config import BotConfig, BotRiskConfig

.. deprecated::
    Removal target: v3.0
    Tracker: docs/DEPRECATIONS.md
"""

import warnings
from typing import Any

# Re-export everything from the new canonical location
from gpt_trader.app.config.bot_config import (
    BotConfig,
    BotRiskConfig,
    MeanReversionConfig,
    StrategyType,
)
from gpt_trader.app.config.defaults import (
    DEFAULT_SPOT_RISK_PATH,
    DEFAULT_SPOT_SYMBOLS,
    TOP_VOLUME_BASES,
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


def __getattr__(name: str) -> Any:
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

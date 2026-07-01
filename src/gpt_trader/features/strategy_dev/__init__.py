"""
Strategy development configuration surface.

The remaining slice provides YAML strategy profiles, a strategy registry, and
the config-diff helpers behind the ``gpt-trader strategy`` CLI command:

    from gpt_trader.features.strategy_dev import (
        ConfigManager,
        StrategyProfile,
        StrategyRegistry,
    )

The former experiment-lab (``lab/``) and performance-monitor (``monitor/``)
subpackages were removed as unwired code; see
``docs/decisions/remove-unwired-account-manager-and-strategy-lab.md``.
"""

from gpt_trader.features.strategy_dev.config import (
    ConfigManager,
    StrategyProfile,
    StrategyRegistry,
)

__all__ = [
    "ConfigManager",
    "StrategyProfile",
    "StrategyRegistry",
]

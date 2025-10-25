"""
Configuration for live trading with schema-driven parsing.

.. deprecated::
    This module has been migrated to bot_v2.orchestration.configuration.risk.
    All imports from this module will be redirected to the modern configuration system.
    The legacy RiskConfig dataclass is still available for backward compatibility but
    should be replaced with bot_v2.orchestration.configuration.RiskConfig.

Migration guide:
    - Replace: from bot_v2.config.live_trade_config import RiskConfig
    - With: from bot_v2.orchestration.configuration import RiskConfig
    - Replace: from bot_v2.config.live_trade_config import RISK_CONFIG_ENV_KEYS
    - With: from bot_v2.orchestration.configuration import RISK_CONFIG_ENV_KEYS
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import for static analyzers only
    from bot_v2.orchestration.configuration import (
        RISK_CONFIG_ENV_ALIASES,
        RISK_CONFIG_ENV_KEYS,
        RiskConfig,
    )

__all__ = [
    "RiskConfig",
    "RISK_CONFIG_ENV_KEYS",
    "RISK_CONFIG_ENV_ALIASES",
    "from_env",
    "from_json",
    "from_legacy_config",
]


def _modern_config_module():
    """Return the modern configuration module."""
    return import_module("bot_v2.orchestration.configuration")


def __getattr__(name: str) -> Any:
    """Redirect deprecated imports to the modern configuration system."""
    if name not in __all__:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            "This module is deprecated. Import from 'bot_v2.orchestration.configuration' instead."
        )

    modern = _modern_config_module()
    attr = getattr(modern, name)
    globals()[name] = attr
    return attr


def __dir__():
    """Return available attributes for tab completion."""
    return __all__ + ["__getattr__", "__dir__", "__all__", "__doc__"]


# Legacy helper functions with deprecation warnings


def from_env(*, settings: Any = None) -> Any:
    """Legacy from_env helper - deprecated.

    Use bot_v2.orchestration.configuration.RiskConfig.from_env() instead.
    """
    warnings.warn(
        "from_env() is deprecated. Use bot_v2.orchestration.configuration.RiskConfig.from_env() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _modern_config_module().RiskConfig.from_env(settings=settings)


def from_json(path: str) -> Any:
    """Legacy from_json helper - deprecated.

    Use bot_v2.orchestration.configuration.RiskConfig.from_json() instead.
    """
    warnings.warn(
        "from_json() is deprecated. Use bot_v2.orchestration.configuration.RiskConfig.from_json() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _modern_config_module().RiskConfig.from_json(path)


def from_legacy_config(legacy_config: Any) -> Any:
    """Legacy from_legacy_config helper - deprecated.

    Use bot_v2.orchestration.configuration.RiskConfig.from_legacy_config() instead.
    """
    warnings.warn(
        "from_legacy_config() is deprecated. Use bot_v2.orchestration.configuration.RiskConfig.from_legacy_config() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _modern_config_module().RiskConfig.from_legacy_config(legacy_config)

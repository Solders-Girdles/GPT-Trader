"""Compatibility shims for legacy configuration schema imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["BotConfigSchema", "ConfigValidationResult"]


if TYPE_CHECKING:
    # During type checking, resolve the symbols eagerly for accurate typing.
    from bot_v2.orchestration.configuration import BotConfig as BotConfigSchema
    from bot_v2.orchestration.configuration import ConfigValidationResult


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)

    from bot_v2.orchestration.configuration import (
        BotConfig as _BotConfig,
    )
    from bot_v2.orchestration.configuration import (
        ConfigValidationResult as _ConfigValidationResult,
    )

    globals()["BotConfigSchema"] = _BotConfig
    globals()["ConfigValidationResult"] = _ConfigValidationResult
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))

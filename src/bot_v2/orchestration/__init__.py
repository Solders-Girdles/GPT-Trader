"""Modern orchestration exports for spot-focused stack.

Provides both legacy bootstrap functions and modern ApplicationContainer-based alternatives.
Prefer the modern functions for new code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time cycle guard
    # Legacy imports (deprecated)
    # Modern container
    from bot_v2.app.container import create_application_container
    from bot_v2.orchestration.bootstrap import bot_from_profile, build_bot

    # Modern configuration
    from bot_v2.orchestration.configuration import BotConfig, ConfigManager, Profile

    # Bot classes
    from bot_v2.orchestration.perps_bot import CoinbaseTrader, PerpsBot

    # Legacy service registry
    from bot_v2.orchestration.service_registry import ServiceRegistry


__all__ = [
    # Legacy functions (deprecated)
    "build_bot",
    "bot_from_profile",
    # Modern functions (recommended)
    "build_bot_modern",
    "bot_from_profile_modern",
    # Core components
    "BotConfig",
    "ConfigManager",
    "create_application_container",
    "Profile",
    "ServiceRegistry",
    "PerpsBot",
    "CoinbaseTrader",
]


def __getattr__(name: str) -> Any:
    """Dynamic attribute resolution with deprecation warnings for legacy functions."""

    # Legacy bootstrap functions (deprecated)
    if name in {"build_bot", "bot_from_profile"}:
        import warnings

        warnings.warn(
            f"{name}() is deprecated. Use build_bot_modern() or bot_from_profile_modern() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from bot_v2.orchestration.bootstrap import bot_from_profile, build_bot

        return build_bot if name == "build_bot" else bot_from_profile

    # Modern bootstrap functions (recommended)
    if name in {"build_bot_modern", "bot_from_profile_modern"}:
        from bot_v2.orchestration.bootstrap import (
            bot_from_profile_modern,
            build_bot_modern,
        )

        return build_bot_modern if name == "build_bot_modern" else bot_from_profile_modern

    # Configuration components
    if name in {"BotConfig", "ConfigManager", "Profile"}:
        from bot_v2.orchestration.configuration import BotConfig, ConfigManager, Profile

        return {"BotConfig": BotConfig, "ConfigManager": ConfigManager, "Profile": Profile}[name]

    # Container factory
    if name == "create_application_container":
        from bot_v2.app.container import create_application_container

        return create_application_container

    # Legacy ServiceRegistry
    if name == "ServiceRegistry":
        import warnings

        warnings.warn(
            "ServiceRegistry is deprecated. Use ApplicationContainer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from bot_v2.orchestration.service_registry import ServiceRegistry

        return ServiceRegistry

    # Bot classes
    if name == "PerpsBot":
        from bot_v2.orchestration.perps_bot import CoinbaseTrader, PerpsBot

        return PerpsBot
    if name == "CoinbaseTrader":
        from bot_v2.orchestration.perps_bot import CoinbaseTrader

        return CoinbaseTrader

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

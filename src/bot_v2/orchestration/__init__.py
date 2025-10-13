"""Modern orchestration exports for the spot-focused stack."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time cycle guard
    from bot_v2.orchestration.bootstrap import bot_from_profile, build_bot
    from bot_v2.orchestration.configuration import BotConfig, ConfigManager, Profile
    from bot_v2.orchestration.perps_bot import PerpsBot
    from bot_v2.orchestration.service_registry import ServiceRegistry

__all__ = [
    "build_bot",
    "bot_from_profile",
    "BotConfig",
    "ConfigManager",
    "Profile",
    "ServiceRegistry",
    "PerpsBot",
]


def __getattr__(name: str) -> Any:
    if name in {"build_bot", "bot_from_profile"}:
        from bot_v2.orchestration.bootstrap import bot_from_profile, build_bot

        return build_bot if name == "build_bot" else bot_from_profile
    if name in {"BotConfig", "ConfigManager", "Profile"}:
        from bot_v2.orchestration.configuration import BotConfig, ConfigManager, Profile

        return {"BotConfig": BotConfig, "ConfigManager": ConfigManager, "Profile": Profile}[name]
    if name == "ServiceRegistry":
        from bot_v2.orchestration.service_registry import ServiceRegistry

        return ServiceRegistry
    if name == "PerpsBot":
        from bot_v2.orchestration.perps_bot import PerpsBot

        return PerpsBot
    raise AttributeError(name)

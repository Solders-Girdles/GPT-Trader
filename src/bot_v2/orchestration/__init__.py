"""Modern orchestration exports for the spot-focused stack."""

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

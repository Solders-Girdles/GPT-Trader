"""Modern orchestration exports for the spot-focused stack."""

from .bootstrap import bot_from_profile, build_bot
from .configuration import BotConfig, ConfigManager, Profile
from .perps_bot import PerpsBot
from .service_registry import ServiceRegistry

__all__ = [
    "build_bot",
    "bot_from_profile",
    "BotConfig",
    "ConfigManager",
    "Profile",
    "ServiceRegistry",
    "PerpsBot",
]

"""Helper utilities for CLI command implementations."""

from argparse import Namespace
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.perps_bot.bot import PerpsBot

def build_config_from_args(args: Namespace, **kwargs) -> BotConfig:
    # Minimal implementation ignoring most kwargs for now
    # In a real simplification, we'd parse args to override env vars
    return BotConfig.from_env()

def instantiate_bot(config: BotConfig) -> PerpsBot:
    """Instantiate a PerpsBot."""
    return PerpsBot(config)

"""Helper utilities for CLI command implementations."""

from argparse import Namespace
from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.orchestration.trading_bot.bot import TradingBot
from gpt_trader.app.container import create_application_container

def build_config_from_args(args: Namespace, **kwargs) -> BotConfig:
    # Minimal implementation ignoring most kwargs for now
    # In a real simplification, we'd parse args to override env vars
    return BotConfig.from_env()

def instantiate_bot(config: BotConfig) -> TradingBot:
    """Instantiate a TradingBot using the ApplicationContainer."""
    container = create_application_container(config)
    return container.create_bot()
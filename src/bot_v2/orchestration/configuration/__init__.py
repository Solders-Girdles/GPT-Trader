"""
Configuration module.
"""
from bot_v2.orchestration.configuration.bot_config import BotConfig, config
from bot_v2.orchestration.configuration.risk.model import RiskConfig

__all__ = ["BotConfig", "config", "RiskConfig"]

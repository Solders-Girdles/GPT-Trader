"""Helper utilities for CLI command implementations."""

from argparse import Namespace
from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.orchestration.trading_bot.bot import TradingBot
from gpt_trader.app.container import create_application_container
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="cli_services")

def build_config_from_args(args: Namespace, **kwargs) -> BotConfig:
    """
    Build configuration from environment, profile, and CLI arguments.
    Precedence: CLI Args > Profile > Environment > Defaults
    """
    import yaml
    from pathlib import Path
    
    # 1. Start with Env/Defaults
    config = BotConfig.from_env()
    
    # 2. Load Profile if specified
    profile_name = getattr(args, "profile", "dev")
    profile_path = Path(f"config/profiles/{profile_name}.yaml")
    
    if profile_path.exists():
        try:
            with open(profile_path) as f:
                profile_data = yaml.safe_load(f)
                
            # Map profile fields to BotConfig
            # Note: BotConfig is flat, Profile is nested. We do best-effort mapping.
            trading = profile_data.get("trading", {})
            if "symbols" in trading:
                config.symbols = trading["symbols"]
            
            # Map other fields as needed/available in BotConfig
            # (BotConfig currently limited, so we only map what matches)
            
        except Exception as e:
            # Log warning but proceed
            logger.warning("Failed to load profile %s: %s", profile_name, e)

    # 3. Override with CLI Args
    if getattr(args, "dry_run", False):
        config.dry_run = True
        
    if getattr(args, "symbols", None):
        config.symbols = args.symbols

    if getattr(args, "interval", None):
        config.interval = args.interval

    return config

def instantiate_bot(config: BotConfig) -> TradingBot:
    """Instantiate a TradingBot using the ApplicationContainer."""
    container = create_application_container(config)
    return container.create_bot()
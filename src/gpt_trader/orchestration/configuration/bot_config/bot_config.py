"""
Simple Bot Configuration.
Replaces the 550-line enterprise configuration system.
"""

import os
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto

# Constants expected by core.py shim
DEFAULT_SPOT_RISK_PATH = "config/risk.json"
DEFAULT_SPOT_SYMBOLS = ["BTC-USD", "ETH-USD"]
TOP_VOLUME_BASES = ["BTC", "ETH", "SOL"]


class ConfigState(Enum):
    INITIALIZED = auto()
    LOADED = auto()
    VALIDATED = auto()
    ERROR = auto()


@dataclass
class BotConfig:
    # Trading Parameters
    max_position_size: Decimal = Decimal("1000")
    max_leverage: int = 5
    stop_loss_pct: Decimal = Decimal("0.02")
    take_profit_pct: Decimal = Decimal("0.04")

    # Strategy Parameters
    short_ma: int = 10
    long_ma: int = 20
    interval: int = 60  # seconds

    # Symbols
    symbols: list[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD"])

    # Derivatives/Perpetuals Parameters
    derivatives_enabled: bool = False
    trailing_stop_pct: Decimal = Decimal("0.01")
    perps_position_fraction: Decimal = Decimal("0.1")
    target_leverage: int = 1
    enable_shorts: bool = False

    # System
    log_level: str = "INFO"
    dry_run: bool = False
    mock_broker: bool = False
    profile: object = None

    @classmethod
    def from_profile(
        cls,
        profile: object,
        dry_run: bool = False,
        mock_broker: bool = False,
        **kwargs: object,
    ) -> "BotConfig":
        """Create a config from a profile name or enum."""
        return cls(profile=profile, dry_run=dry_run, mock_broker=mock_broker)

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Load configuration from environment variables."""
        from gpt_trader.config.config_utilities import (
            parse_bool_env,
            parse_decimal_env,
            parse_int_env,
            parse_list_env,
        )

        return cls(
            max_position_size=parse_decimal_env("MAX_POSITION_SIZE", Decimal("1000")),
            max_leverage=parse_int_env("MAX_LEVERAGE", 5),
            stop_loss_pct=parse_decimal_env("STOP_LOSS_PCT", Decimal("0.02")),
            take_profit_pct=parse_decimal_env("TAKE_PROFIT_PCT", Decimal("0.04")),
            short_ma=parse_int_env("SHORT_MA", 10),
            long_ma=parse_int_env("LONG_MA", 20),
            interval=parse_int_env("INTERVAL", 60),
            symbols=parse_list_env("SYMBOLS", str, default=["BTC-USD", "ETH-USD"]),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            dry_run=parse_bool_env("DRY_RUN", default=False),
        )


# Global config instance
config = BotConfig.from_env()

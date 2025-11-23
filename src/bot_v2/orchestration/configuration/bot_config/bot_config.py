"""
Simple Bot Configuration.
Replaces the 550-line enterprise configuration system.
"""
import os
from dataclasses import dataclass, field
from decimal import Decimal

@dataclass
class BotConfig:
    # Trading Parameters
    max_position_size: Decimal = Decimal("1000")
    stop_loss_pct: Decimal = Decimal("0.02")
    take_profit_pct: Decimal = Decimal("0.04")

    # Strategy Parameters
    short_ma: int = 10
    long_ma: int = 20
    interval: int = 60  # seconds

    # Symbols
    symbols: list[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD"])

    # System
    log_level: str = "INFO"
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Load configuration from environment variables."""
        return cls(
            max_position_size=Decimal(os.getenv("MAX_POSITION_SIZE", "1000")),
            stop_loss_pct=Decimal(os.getenv("STOP_LOSS_PCT", "0.02")),
            take_profit_pct=Decimal(os.getenv("TAKE_PROFIT_PCT", "0.04")),
            short_ma=int(os.getenv("SHORT_MA", "10")),
            long_ma=int(os.getenv("LONG_MA", "20")),
            interval=int(os.getenv("INTERVAL", "60")),
            symbols=os.getenv("SYMBOLS", "BTC-USD,ETH-USD").split(","),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            dry_run=os.getenv("DRY_RUN", "False").lower() == "true",
        )

# Global config instance
config = BotConfig.from_env()

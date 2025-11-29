"""
Simple Bot Configuration.
Replaces the 550-line enterprise configuration system.

Supports nested configuration structure for optimization framework compatibility:
- strategy: Trading strategy parameters (uses PerpsStrategyConfig)
- risk: Risk management parameters (uses RiskConfig)
"""

import os
from dataclasses import dataclass, field, fields, replace
from decimal import Decimal
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal

# Constants expected by core.py shim
DEFAULT_SPOT_RISK_PATH = "config/risk.json"
DEFAULT_SPOT_SYMBOLS = ["BTC-USD", "ETH-USD"]
TOP_VOLUME_BASES = ["BTC", "ETH", "SOL"]

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.strategies.perps_baseline import (
        PerpsStrategyConfig,
    )


class ConfigState(Enum):
    INITIALIZED = auto()
    LOADED = auto()
    VALIDATED = auto()
    ERROR = auto()


@dataclass
class BotRiskConfig:
    """Bot-level risk/position sizing configuration.

    Holds position sizing, leverage, and loss control parameters.
    Compatible with optimization framework 'risk' section output.

    Note: Distinct from risk.model.RiskConfig which is for the risk manager.
    """

    target_leverage: int = 1
    max_leverage: int = 5
    position_fraction: Decimal = Decimal("0.1")
    max_position_size: Decimal = Decimal("1000")
    stop_loss_pct: Decimal = Decimal("0.02")
    take_profit_pct: Decimal = Decimal("0.04")
    trailing_stop_pct: Decimal = Decimal("0.01")
    max_drawdown_pct: Decimal | None = None
    reduce_only_threshold: Decimal | None = None


@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion strategy.

    Uses Z-Score based entries with volatility-targeted position sizing.
    """

    # Z-Score thresholds for entry/exit
    z_score_entry_threshold: float = 2.0  # Enter when |z-score| > this
    z_score_exit_threshold: float = 0.5  # Exit when |z-score| < this (near mean)

    # Lookback window for rolling statistics
    lookback_window: int = 20

    # Volatility targeting for position sizing
    target_daily_volatility: float = 0.02  # 2% target daily vol
    max_position_pct: float = 0.25  # Max 25% of equity per position

    # Risk controls
    stop_loss_pct: float = 0.03  # 3% stop loss
    take_profit_pct: float = 0.06  # 6% take profit

    # Control flags
    enable_shorts: bool = True
    kill_switch_enabled: bool = False


# Strategy type literal for type safety
StrategyType = Literal["baseline", "mean_reversion"]


def _get_default_strategy_config() -> "PerpsStrategyConfig":
    """Lazy factory to avoid circular imports."""
    from gpt_trader.features.live_trade.strategies.perps_baseline import (
        PerpsStrategyConfig,
    )

    return PerpsStrategyConfig()


@dataclass
class BotConfig:
    """Bot configuration with nested strategy and risk configs.

    Supports both:
    - Nested access: config.strategy.short_ma_period, config.risk.max_leverage
    - Flat access (backward compat): config.short_ma, config.max_leverage
    """

    # Nested configurations
    strategy: Any = field(default_factory=_get_default_strategy_config)
    risk: BotRiskConfig = field(default_factory=BotRiskConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)

    # Strategy selection (baseline = RSI+MA, mean_reversion = Z-Score)
    strategy_type: StrategyType = "baseline"

    # General config (not nested)
    symbols: list[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD"])
    interval: int = 60  # seconds
    derivatives_enabled: bool = False
    enable_shorts: bool = False
    reduce_only_mode: bool = False
    time_in_force: str = "GTC"
    enable_order_preview: bool = False
    account_telemetry_interval: int | None = None

    # System
    log_level: str = "INFO"
    dry_run: bool = False
    mock_broker: bool = False
    profile: object = None

    # Notifications (Slack/Discord webhook URL)
    webhook_url: str | None = None

    # Operational monitoring
    status_file: str = "var/data/status.json"
    status_interval: int = 60  # Seconds between status file updates
    status_enabled: bool = True

    # Metadata (Pydantic compatibility)
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- Backward-compatible properties for flat access ---

    # Strategy field aliases
    @property
    def short_ma(self) -> int:
        """Alias for strategy.short_ma_period (backward compat)."""
        return self.strategy.short_ma_period

    @property
    def long_ma(self) -> int:
        """Alias for strategy.long_ma_period (backward compat)."""
        return self.strategy.long_ma_period

    # Risk field aliases
    @property
    def max_position_size(self) -> Decimal:
        """Alias for risk.max_position_size (backward compat)."""
        return self.risk.max_position_size

    @property
    def max_leverage(self) -> int:
        """Alias for risk.max_leverage (backward compat)."""
        return self.risk.max_leverage

    @property
    def target_leverage(self) -> int:
        """Alias for risk.target_leverage (backward compat)."""
        return self.risk.target_leverage

    @property
    def stop_loss_pct(self) -> Decimal:
        """Alias for risk.stop_loss_pct (backward compat)."""
        return self.risk.stop_loss_pct

    @property
    def take_profit_pct(self) -> Decimal:
        """Alias for risk.take_profit_pct (backward compat)."""
        return self.risk.take_profit_pct

    @property
    def trailing_stop_pct(self) -> Decimal:
        """Alias for risk.trailing_stop_pct (backward compat)."""
        return self.risk.trailing_stop_pct

    @property
    def perps_position_fraction(self) -> Decimal:
        """Alias for risk.position_fraction (backward compat)."""
        return self.risk.position_fraction

    @property
    def model_fields(self) -> dict[str, Any]:
        """Pydantic-compatible property to list field names."""
        return {f.name: f for f in fields(self)}

    def model_copy(self, update: dict[str, Any] | None = None) -> "BotConfig":
        """Pydantic-compatible method to create a copy with updates."""
        if update is None:
            return replace(self)
        return replace(self, **update)

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
        """Load configuration from environment variables.

        Creates nested strategy and risk configs from env vars.
        """
        from gpt_trader.config.config_utilities import (
            parse_bool_env,
            parse_decimal_env,
            parse_int_env,
            parse_list_env,
        )
        from gpt_trader.features.live_trade.strategies.perps_baseline import (
            PerpsStrategyConfig,
        )

        # Build strategy config from env
        strategy = PerpsStrategyConfig(
            short_ma_period=parse_int_env("SHORT_MA", 5) or 5,
            long_ma_period=parse_int_env("LONG_MA", 20) or 20,
        )

        # Build risk config from env
        risk = BotRiskConfig(
            max_position_size=parse_decimal_env("MAX_POSITION_SIZE", Decimal("1000"))
            or Decimal("1000"),
            max_leverage=parse_int_env("MAX_LEVERAGE", 5) or 5,
            target_leverage=parse_int_env("TARGET_LEVERAGE", 1) or 1,
            stop_loss_pct=parse_decimal_env("STOP_LOSS_PCT", Decimal("0.02")) or Decimal("0.02"),
            take_profit_pct=parse_decimal_env("TAKE_PROFIT_PCT", Decimal("0.04"))
            or Decimal("0.04"),
            position_fraction=parse_decimal_env("POSITION_FRACTION", Decimal("0.1"))
            or Decimal("0.1"),
        )

        return cls(
            strategy=strategy,
            risk=risk,
            interval=parse_int_env("INTERVAL", 60) or 60,
            symbols=parse_list_env("SYMBOLS", str, default=["BTC-USD", "ETH-USD"]),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            dry_run=parse_bool_env("DRY_RUN", default=False),
            webhook_url=os.getenv("WEBHOOK_URL"),
            status_file=os.getenv("STATUS_FILE", "var/data/status.json"),
            status_interval=parse_int_env("STATUS_INTERVAL", 60) or 60,
            status_enabled=parse_bool_env("STATUS_ENABLED", default=True),
        )


# Global config instance
config = BotConfig.from_env()

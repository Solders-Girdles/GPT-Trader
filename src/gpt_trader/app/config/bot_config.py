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
from typing import TYPE_CHECKING, Any, ClassVar, Literal

# Constants expected by core.py shim
DEFAULT_SPOT_RISK_PATH = "config/risk.json"
DEFAULT_SPOT_SYMBOLS = ["BTC-USD", "ETH-USD"]
TOP_VOLUME_BASES = ["BTC", "ETH", "SOL"]

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.strategies.perps_baseline import (
        PerpsStrategyConfig,
    )


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
    # Daily loss limit as percentage of equity (0.05 = 5%)
    # When daily loss exceeds this, reduce-only mode is triggered
    daily_loss_limit_pct: float = 0.05


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
StrategyType = Literal["baseline", "mean_reversion", "ensemble"]


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

    # Intelligence feature configurations (optional, for ensemble strategy)
    regime_config: Any = None  # RegimeConfig instance when using ensemble
    ensemble_config: Any = None  # EnsembleConfig instance when using ensemble

    # Strategy selection (baseline = RSI+MA, mean_reversion = Z-Score, ensemble = multi-strategy)
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

    # Broker / Exchange Settings
    coinbase_default_quote: str = "USD"
    coinbase_us_futures_enabled: bool = False
    coinbase_intx_perpetuals_enabled: bool = False
    coinbase_derivatives_type: str = "intx_perps"
    coinbase_sandbox_enabled: bool = False
    coinbase_api_mode: str = "advanced"
    coinbase_intx_portfolio_uuid: str | None = None
    broker_hint: str | None = None

    # CFM (Coinbase Financial Markets) Futures Settings
    # trading_modes: Which markets to trade in - "spot", "cfm", or both for hybrid
    trading_modes: list[str] = field(default_factory=lambda: ["spot"])
    cfm_enabled: bool = False  # Explicit CFM futures flag
    cfm_max_leverage: int = 5  # Maximum leverage for CFM positions
    cfm_symbols: list[str] = field(default_factory=list)  # CFM-specific symbols
    cfm_margin_window: str = "STANDARD"  # STANDARD, INTRADAY_STANDARD, INTRADAY_PLUS

    # Perps / Futures Specifics
    perps_enable_streaming: bool = False
    perps_stream_level: int = 1
    perps_paper_trading: bool = False
    perps_skip_startup_reconcile: bool = False
    perps_position_fraction: float | None = None

    # Paths & Environments
    runtime_root: str = "."
    event_store_root_override: str | None = None
    risk_config_path: str | None = None
    environment: str | None = None
    spot_force_live: bool = False

    # Metadata (Pydantic compatibility)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_fields(self) -> dict[str, Any]:
        """Pydantic-compatible property to list field names."""
        return {f.name: f for f in fields(self)}

    def model_copy(self, update: dict[str, Any] | None = None) -> "BotConfig":
        """Pydantic-compatible method to create a copy with updates."""
        if update is None:
            return replace(self)
        return replace(self, **update)

    @property
    def is_hybrid_mode(self) -> bool:
        """Check if trading in hybrid mode (both spot and CFM)."""
        return "spot" in self.trading_modes and "cfm" in self.trading_modes

    @property
    def is_cfm_only(self) -> bool:
        """Check if trading CFM futures only (no spot)."""
        return "cfm" in self.trading_modes and "spot" not in self.trading_modes

    @property
    def is_spot_only(self) -> bool:
        """Check if trading spot only (no CFM)."""
        return "spot" in self.trading_modes and "cfm" not in self.trading_modes

    @property
    def active_enable_shorts(self) -> bool:
        """Get enable_shorts from active strategy config (canonical source).

        Derives enable_shorts from the strategy config based on strategy_type.
        Warns once if top-level enable_shorts differs from strategy config.
        """
        import warnings

        # Get canonical value from active strategy
        if self.strategy_type == "mean_reversion":
            canonical = self.mean_reversion.enable_shorts
        else:
            # baseline, ensemble use strategy config
            canonical = getattr(self.strategy, "enable_shorts", False)

        # Warn on mismatch (once per process)
        if self.enable_shorts != canonical and not BotConfig._enable_shorts_sync_warned:
            warnings.warn(
                f"BotConfig.enable_shorts={self.enable_shorts} differs from "
                f"strategy config ({canonical}). Strategy config is canonical.",
                UserWarning,
                stacklevel=2,
            )
            BotConfig._enable_shorts_sync_warned = True

        return canonical

    @classmethod
    def from_profile(
        cls,
        profile: object,
        dry_run: bool = False,
        mock_broker: bool = False,
        **kwargs: object,
    ) -> "BotConfig":
        """Create a config from a profile name or enum."""
        # Type ignore: kwargs unpacking is dynamic, but we trust the caller to provide valid fields
        return cls(profile=profile, dry_run=dry_run, mock_broker=mock_broker, **kwargs)  # type: ignore[arg-type]

    # Single-shot deprecation warning for PERPS_FORCE_MOCK
    _perps_force_mock_warned: ClassVar[bool] = False
    # Single-shot sync warning for enable_shorts mismatch
    _enable_shorts_sync_warned: ClassVar[bool] = False

    @classmethod
    def _parse_mock_broker_env(cls) -> bool:
        """Parse mock_broker from env with deprecated PERPS_FORCE_MOCK fallback.

        Precedence: MOCK_BROKER > PERPS_FORCE_MOCK (deprecated)
        """
        import warnings

        from gpt_trader.config.config_utilities import parse_bool_env

        # Check canonical env var first
        mock_broker_env = os.getenv("MOCK_BROKER")
        if mock_broker_env is not None:
            return parse_bool_env("MOCK_BROKER", default=False)

        # Check deprecated alias
        perps_force_mock = os.getenv("PERPS_FORCE_MOCK")
        if perps_force_mock is not None:
            if not cls._perps_force_mock_warned:
                warnings.warn(
                    "PERPS_FORCE_MOCK is deprecated. Use MOCK_BROKER=1 instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                cls._perps_force_mock_warned = True
            return perps_force_mock.lower() in ("1", "true", "yes")

        return False

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

        # Build risk config from env (supports RISK_* prefix for all fields)
        # RISK_* prefixed names take precedence over legacy unprefixed names
        def _risk_env(key: str, default: str) -> str:
            """Get risk env var with RISK_ prefix taking precedence."""
            prefixed = os.getenv(f"RISK_{key}")
            if prefixed is not None:
                return prefixed
            return os.getenv(key, default)

        def _risk_int(key: str, default: int) -> int:
            """Get risk int env var with RISK_ prefix taking precedence."""
            return int(_risk_env(key, str(default)))

        def _risk_decimal(key: str, default: str) -> Decimal:
            """Get risk Decimal env var with RISK_ prefix taking precedence."""
            return Decimal(_risk_env(key, default))

        def _risk_float(key: str, default: float) -> float:
            """Get risk float env var with RISK_ prefix taking precedence."""
            return float(_risk_env(key, str(default)))

        # Map RISK_MAX_POSITION_PCT_PER_SYMBOL -> position_fraction
        # Also support legacy POSITION_FRACTION
        position_fraction_raw = (
            os.getenv("RISK_MAX_POSITION_PCT_PER_SYMBOL") or os.getenv("POSITION_FRACTION") or "0.1"
        )

        risk = BotRiskConfig(
            max_position_size=_risk_decimal("MAX_POSITION_SIZE", "1000"),
            max_leverage=_risk_int("MAX_LEVERAGE", 5),
            target_leverage=_risk_int("TARGET_LEVERAGE", 1),
            stop_loss_pct=_risk_decimal("STOP_LOSS_PCT", "0.02"),
            take_profit_pct=_risk_decimal("TAKE_PROFIT_PCT", "0.04"),
            position_fraction=Decimal(position_fraction_raw),
            # Daily loss limit as percentage (0.05 = 5%)
            daily_loss_limit_pct=_risk_float("DAILY_LOSS_LIMIT_PCT", 0.05),
        )

        derivatives_enabled = parse_bool_env("COINBASE_ENABLE_DERIVATIVES", default=False)

        # Support both TRADING_SYMBOLS (canonical) and SYMBOLS (legacy)
        symbols_raw = os.getenv("TRADING_SYMBOLS") or os.getenv("SYMBOLS")
        if symbols_raw:
            symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]
        else:
            symbols = ["BTC-USD", "ETH-USD"]

        return cls(
            strategy=strategy,
            risk=risk,
            interval=parse_int_env("INTERVAL", 60) or 60,
            symbols=symbols,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            dry_run=parse_bool_env("DRY_RUN", default=False),
            webhook_url=os.getenv("WEBHOOK_URL"),
            status_file=os.getenv("STATUS_FILE", "var/data/status.json"),
            status_interval=parse_int_env("STATUS_INTERVAL", 60) or 60,
            status_enabled=parse_bool_env("STATUS_ENABLED", default=True),
            derivatives_enabled=derivatives_enabled,
            # Inherited from RuntimeSettings
            coinbase_default_quote=os.getenv("COINBASE_DEFAULT_QUOTE", "USD"),
            coinbase_us_futures_enabled=parse_bool_env(
                "COINBASE_US_FUTURES_ENABLED", default=False
            ),
            coinbase_intx_perpetuals_enabled=parse_bool_env(
                "COINBASE_INTX_PERPETUALS_ENABLED", default=False
            ),
            coinbase_derivatives_type=os.getenv("COINBASE_DERIVATIVES_TYPE", "intx_perps"),
            coinbase_sandbox_enabled=parse_bool_env("COINBASE_SANDBOX", default=False),
            coinbase_api_mode=os.getenv("COINBASE_API_MODE", "advanced"),
            coinbase_intx_portfolio_uuid=os.getenv("COINBASE_INTX_PORTFOLIO_UUID"),
            broker_hint=os.getenv("BROKER"),
            # CFM settings
            trading_modes=parse_list_env("TRADING_MODES", str, default=["spot"]),
            cfm_enabled=parse_bool_env("CFM_ENABLED", default=False),
            cfm_max_leverage=parse_int_env("CFM_MAX_LEVERAGE", 5) or 5,
            cfm_symbols=parse_list_env("CFM_SYMBOLS", str, default=[]),
            cfm_margin_window=os.getenv("CFM_MARGIN_WINDOW", "STANDARD"),
            perps_enable_streaming=parse_bool_env("PERPS_ENABLE_STREAMING", default=False),
            perps_stream_level=parse_int_env("PERPS_STREAM_LEVEL", 1) or 1,
            perps_paper_trading=parse_bool_env("PERPS_PAPER", default=False),
            perps_skip_startup_reconcile=parse_bool_env("PERPS_SKIP_RECONCILE", default=False),
            perps_position_fraction=float(
                parse_decimal_env("PERPS_POSITION_FRACTION", Decimal("0.1")) or Decimal("0.1")
            ),
            runtime_root=os.getenv("GPT_TRADER_RUNTIME_ROOT", "."),
            event_store_root_override=os.getenv("EVENT_STORE_ROOT"),
            risk_config_path=os.getenv("RISK_CONFIG_PATH"),
            environment=os.getenv("ENVIRONMENT"),
            spot_force_live=parse_bool_env("SPOT_FORCE_LIVE", default=False),
            enable_order_preview=parse_bool_env("ORDER_PREVIEW_ENABLED", default=False),
            # Risk modes from env (CLI/profile override these)
            reduce_only_mode=parse_bool_env(
                "RISK_REDUCE_ONLY_MODE",
                default=parse_bool_env("REDUCE_ONLY_MODE", default=False),
            ),
            # Mock broker with deprecated PERPS_FORCE_MOCK fallback
            mock_broker=cls._parse_mock_broker_env(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BotConfig":
        """Create configuration from a dictionary (e.g. loaded from YAML).

        Handles mapping from legacy profile structure to BotConfig structure.
        """
        from gpt_trader.features.live_trade.strategies.perps_baseline import (
            PerpsStrategyConfig,
        )

        # 1. Map top-level fields
        config_data = {}

        # Direct mappings
        if "symbols" in data:
            config_data["symbols"] = data["symbols"]
        if "paper_mode" in data:
            config_data["perps_paper_trading"] = data["paper_mode"]
        if "log_level" in data:
            config_data["log_level"] = data["log_level"]

        # 2. Map Risk Config
        risk_data: dict[str, Any] = {}
        source_risk = data.get("risk", {})

        # Map legacy risk keys to BotRiskConfig keys
        if "max_position_pct" in source_risk:
            risk_data["position_fraction"] = Decimal(str(source_risk["max_position_pct"]))
        if "max_leverage" in source_risk:
            risk_data["max_leverage"] = int(source_risk["max_leverage"])
        if "stop_loss_pct" in source_risk:
            risk_data["stop_loss_pct"] = Decimal(str(source_risk["stop_loss_pct"]))
        if "take_profit_pct" in source_risk:
            risk_data["take_profit_pct"] = Decimal(str(source_risk["take_profit_pct"]))

        # Explicit cast to fix mypy assignment error
        if "max_position_size" in source_risk:
            risk_data["max_position_size"] = Decimal(str(source_risk["max_position_size"]))

        # Filter unexpected keys to avoid TypeError
        valid_risk_fields = {f.name for f in fields(BotRiskConfig)}
        filtered_risk_data = {k: v for k, v in risk_data.items() if k in valid_risk_fields}

        config_data["risk"] = BotRiskConfig(**filtered_risk_data)

        # 3. Map Execution Config
        execution = data.get("execution", {})
        if "mock_broker" in execution:
            config_data["mock_broker"] = execution["mock_broker"]
        if "dry_run" in execution:
            config_data["dry_run"] = execution["dry_run"]

        # 3. Map Strategy Config
        # Legacy profiles often have strategy nested by symbol (e.g. strategy.btc)
        # We'll take the first available strategy config or a default
        source_strategy = data.get("strategy", {})
        strategy_config_data = {}

        # Flatten symbol-specific strategy if present
        first_strategy: dict[str, Any] = (
            next(iter(source_strategy.values()), {}) if source_strategy else {}
        )
        if isinstance(first_strategy, dict) and "short_window" in first_strategy:
            # It's a legacy strategy dict
            strategy_config_data["short_ma_period"] = first_strategy.get("short_window", 5)
            strategy_config_data["long_ma_period"] = first_strategy.get("long_window", 20)

            # Map filter configs if needed, or just basic params for now

        config_data["strategy"] = PerpsStrategyConfig(**strategy_config_data)

        return cls(**config_data)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "BotConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

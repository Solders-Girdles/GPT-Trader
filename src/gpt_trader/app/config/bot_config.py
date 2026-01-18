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

# Import canonical defaults (top-10 symbols).

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.strategies.perps_baseline import (
        PerpsStrategyConfig,
    )
    from gpt_trader.monitoring.health_signals import HealthThresholds


@dataclass
class HealthThresholdsConfig:
    """Configurable thresholds for health signals.

    All thresholds use sensible defaults aligned with HealthThresholds model.
    Set via HEALTH_* env vars or config file.
    """

    # Order submission error rate (failures / total in window)
    order_error_rate_warn: float = 0.05  # 5% warning
    order_error_rate_crit: float = 0.15  # 15% critical

    # Order retry rate (retries / total in window)
    order_retry_rate_warn: float = 0.10  # 10% warning
    order_retry_rate_crit: float = 0.25  # 25% critical

    # Broker API latency p95 (milliseconds)
    broker_latency_ms_warn: float = 1000.0  # 1 second warning
    broker_latency_ms_crit: float = 3000.0  # 3 seconds critical

    # WebSocket staleness (seconds since last message)
    ws_staleness_seconds_warn: float = 30.0  # 30 seconds warning
    ws_staleness_seconds_crit: float = 60.0  # 60 seconds critical

    # Guard trip frequency (trips in window)
    guard_trip_count_warn: int = 3  # 3 trips warning
    guard_trip_count_crit: int = 10  # 10 trips critical

    def to_health_thresholds(self) -> "HealthThresholds":
        """Convert to monitoring.health_signals.HealthThresholds."""
        from gpt_trader.monitoring.health_signals import HealthThresholds

        return HealthThresholds(
            order_error_rate_warn=self.order_error_rate_warn,
            order_error_rate_crit=self.order_error_rate_crit,
            order_retry_rate_warn=self.order_retry_rate_warn,
            order_retry_rate_crit=self.order_retry_rate_crit,
            broker_latency_ms_warn=self.broker_latency_ms_warn,
            broker_latency_ms_crit=self.broker_latency_ms_crit,
            ws_staleness_seconds_warn=self.ws_staleness_seconds_warn,
            ws_staleness_seconds_crit=self.ws_staleness_seconds_crit,
            guard_trip_count_warn=self.guard_trip_count_warn,
            guard_trip_count_crit=self.guard_trip_count_crit,
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

    # Cooldown controls
    cooldown_bars: int = 0

    # Trend filter controls
    trend_filter_enabled: bool = False
    trend_window: int = 50
    trend_threshold_pct: float = 0.01
    trend_override_z_score: float | None = None


# Strategy type literal for type safety
StrategyType = Literal["baseline", "mean_reversion", "ensemble", "regime_switcher"]


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
    health_thresholds: HealthThresholdsConfig = field(default_factory=HealthThresholdsConfig)

    # Intelligence feature configurations (optional, for ensemble strategy)
    regime_config: Any = None  # RegimeConfig instance when using ensemble
    ensemble_config: Any = None  # EnsembleConfig instance when using ensemble

    # Strategy selection
    # - baseline = RSI+MA crossover
    # - mean_reversion = Z-Score
    # - ensemble = multi-signal architecture
    # - regime_switcher = switch between trend and mean reversion by regime
    strategy_type: StrategyType = "baseline"

    regime_switcher_trend_mode: Literal["delegate", "regime_follow"] = "delegate"

    # General config (not nested)
    symbols: list[str] = field(default_factory=lambda: ["BTC-USD", "ETH-USD"])
    interval: int = 60  # seconds
    derivatives_enabled: bool = False
    enable_shorts: bool = False
    reduce_only_mode: bool = False
    time_in_force: str = "GTC"
    enable_order_preview: bool = False
    use_limit_orders: bool = False
    market_order_fallback: bool = True
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

    @property
    def short_ma(self) -> int:
        """Backward-compatible alias for strategy.short_ma_period."""
        return int(getattr(self.strategy, "short_ma_period", 5))

    @property
    def long_ma(self) -> int:
        """Backward-compatible alias for strategy.long_ma_period."""
        return int(getattr(self.strategy, "long_ma_period", 20))

    @property
    def target_leverage(self) -> int:
        """Backward-compatible alias for risk.target_leverage."""
        return int(self.risk.target_leverage)

    @property
    def max_leverage(self) -> int:
        """Backward-compatible alias for risk.max_leverage."""
        return int(self.risk.max_leverage)

    @property
    def trailing_stop_pct(self) -> float | None:
        """Backward-compatible alias for strategy/risk trailing stop."""
        strategy_value = getattr(self.strategy, "trailing_stop_pct", None)
        if strategy_value is not None:
            return float(strategy_value)
        return float(self.risk.trailing_stop_pct)

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

    # Single-shot sync warning for enable_shorts mismatch
    _enable_shorts_sync_warned: ClassVar[bool] = False

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

        # Build risk config from env (RISK_* prefixed fields only)
        def _risk_env(key: str, default: str) -> str:
            """Get risk env var from RISK_ prefixed key."""
            return os.getenv(f"RISK_{key}", default)

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
        position_fraction_raw = os.getenv("RISK_MAX_POSITION_PCT_PER_SYMBOL", "0.1")

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

        intx_perps_raw = os.getenv("COINBASE_ENABLE_INTX_PERPS")
        if intx_perps_raw is not None and intx_perps_raw != "":
            derivatives_enabled = parse_bool_env("COINBASE_ENABLE_INTX_PERPS", default=False)
        else:
            derivatives_enabled = parse_bool_env("COINBASE_ENABLE_DERIVATIVES", default=False)

        # Build health thresholds config from env (HEALTH_* prefix)
        def _health_float(key: str, default: float) -> float:
            """Get health threshold float env var with HEALTH_ prefix."""
            val = os.getenv(f"HEALTH_{key}")
            return float(val) if val is not None else default

        def _health_int(key: str, default: int) -> int:
            """Get health threshold int env var with HEALTH_ prefix."""
            val = os.getenv(f"HEALTH_{key}")
            return int(val) if val is not None else default

        health_thresholds = HealthThresholdsConfig(
            order_error_rate_warn=_health_float("ORDER_ERROR_RATE_WARN", 0.05),
            order_error_rate_crit=_health_float("ORDER_ERROR_RATE_CRIT", 0.15),
            order_retry_rate_warn=_health_float("ORDER_RETRY_RATE_WARN", 0.10),
            order_retry_rate_crit=_health_float("ORDER_RETRY_RATE_CRIT", 0.25),
            broker_latency_ms_warn=_health_float("BROKER_LATENCY_MS_WARN", 1000.0),
            broker_latency_ms_crit=_health_float("BROKER_LATENCY_MS_CRIT", 3000.0),
            ws_staleness_seconds_warn=_health_float("WS_STALENESS_SECONDS_WARN", 30.0),
            ws_staleness_seconds_crit=_health_float("WS_STALENESS_SECONDS_CRIT", 60.0),
            guard_trip_count_warn=_health_int("GUARD_TRIP_COUNT_WARN", 3),
            guard_trip_count_crit=_health_int("GUARD_TRIP_COUNT_CRIT", 10),
        )

        symbols_raw = os.getenv("TRADING_SYMBOLS")
        if symbols_raw:
            symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]
        else:
            symbols = ["BTC-USD", "ETH-USD"]

        return cls(
            strategy=strategy,
            risk=risk,
            health_thresholds=health_thresholds,
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
            reduce_only_mode=parse_bool_env("RISK_REDUCE_ONLY_MODE", default=False),
            mock_broker=parse_bool_env("MOCK_BROKER", default=False),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BotConfig":
        """Create configuration from a dictionary (e.g. loaded from YAML).

        Supports both BotConfig-style and profile-style YAML schemas.
        """
        from dataclasses import fields as dataclass_fields

        from gpt_trader.features.live_trade.strategies.perps_baseline import (
            PerpsStrategyConfig,
        )

        def _filter_dataclass_fields(
            source: dict[str, Any], dataclass_type: type
        ) -> dict[str, Any]:
            valid_fields = {f.name for f in dataclass_fields(dataclass_type)}
            return {k: v for k, v in source.items() if k in valid_fields}

        def _coerce_decimal_fields(source: dict[str, Any], dataclass_type: type) -> dict[str, Any]:
            result = dict(source)
            for f in dataclass_fields(dataclass_type):
                if f.name in result and f.type in (Decimal, "Decimal"):
                    val = result[f.name]
                    if val is not None and not isinstance(val, Decimal):
                        result[f.name] = Decimal(str(val))
            return result

        if any(key in data for key in ("trading", "risk_management", "profile_name")):
            from gpt_trader.app.config.profile_loader import ProfileSchema

            schema = ProfileSchema.from_yaml(data, data.get("profile_name", "custom"))

            risk = BotRiskConfig(
                max_leverage=schema.risk.max_leverage,
                max_position_size=schema.risk.max_position_size,
                position_fraction=schema.risk.position_fraction,
                stop_loss_pct=schema.risk.stop_loss_pct,
                take_profit_pct=schema.risk.take_profit_pct,
                daily_loss_limit_pct=schema.risk.daily_loss_limit_pct,
            )

            strategy = PerpsStrategyConfig(
                short_ma_period=schema.strategy.short_ma_period,
                long_ma_period=schema.strategy.long_ma_period,
                rsi_period=schema.strategy.rsi_period,
                rsi_overbought=schema.strategy.rsi_overbought,
                rsi_oversold=schema.strategy.rsi_oversold,
            )

            config_data: dict[str, Any] = {
                "profile": schema.profile_name or None,
                "environment": schema.environment,
                "symbols": schema.trading.symbols,
                "interval": schema.trading.interval,
                "risk": risk,
                "strategy": strategy,
                "enable_shorts": schema.risk.enable_shorts,
                "time_in_force": schema.execution.time_in_force,
                "dry_run": schema.execution.dry_run,
                "mock_broker": schema.execution.mock_broker,
                "log_level": schema.monitoring.log_level,
                "status_interval": schema.monitoring.update_interval,
                "status_enabled": schema.monitoring.status_enabled,
                "strategy_type": schema.strategy.type,
            }

            if schema.trading.mode == "reduce_only":
                config_data["reduce_only_mode"] = True

            return cls(**config_data)

        strategy_data = data.get("strategy", {})
        if not isinstance(strategy_data, dict):
            strategy_data = {}
        strategy_kwargs = _filter_dataclass_fields(strategy_data, PerpsStrategyConfig)
        strategy = PerpsStrategyConfig(**strategy_kwargs)

        risk_data = data.get("risk", {})
        if not isinstance(risk_data, dict):
            risk_data = {}
        risk_kwargs = _filter_dataclass_fields(risk_data, BotRiskConfig)
        risk_kwargs = _coerce_decimal_fields(risk_kwargs, BotRiskConfig)
        risk = BotRiskConfig(**risk_kwargs)

        valid_bot_fields = {f.name for f in dataclass_fields(cls)}
        config_data = {k: v for k, v in data.items() if k in valid_bot_fields}
        config_data.update({"strategy": strategy, "risk": risk})

        return cls(**config_data)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "BotConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

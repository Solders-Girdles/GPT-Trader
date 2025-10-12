"""Configuration primitives for GPT-Trader orchestration."""

from __future__ import annotations

import logging
from copy import deepcopy
from datetime import time
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic_core import PydanticCustomError

from bot_v2.config import path_registry
from bot_v2.config.types import Profile
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.orchestration.symbols import (
    derivatives_enabled as _resolve_derivatives_enabled,
)
from bot_v2.orchestration.symbols import (
    normalize_symbol_list,
)

logger = logging.getLogger(__name__)

# Top spot markets we enable by default (ordered by Coinbase USD volume).
TOP_VOLUME_BASES = [
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "LTC",
    "ADA",
    "DOGE",
    "BCH",
    "AVAX",
    "LINK",
]

DEFAULT_SPOT_SYMBOLS = [f"{base}-USD" for base in TOP_VOLUME_BASES]

DEFAULT_SPOT_RISK_PATH = Path(__file__).resolve().parents[3] / "config" / "risk" / "spot_top10.json"


class BotConfig(BaseModel):
    """Bot configuration backed by Pydantic validation."""

    profile: Profile
    dry_run: bool = False
    symbols: list[str] | None = None
    derivatives_enabled: bool = False
    update_interval: int = 5
    short_ma: int = 5
    long_ma: int = 20
    target_leverage: int = 2
    trailing_stop_pct: float = 0.01
    enable_shorts: bool = False
    max_position_size: Decimal = Decimal("1000")
    max_leverage: int = 3
    reduce_only_mode: bool = False
    mock_broker: bool = False
    mock_fills: bool = False
    enable_order_preview: bool = False
    account_telemetry_interval: int = 300
    trading_window_start: time | None = None
    trading_window_end: time | None = None
    trading_days: list[str] | None = None
    daily_loss_limit: Decimal = Decimal("0")
    time_in_force: str = "GTC"
    perps_enable_streaming: bool = False
    perps_stream_level: int = 1
    perps_paper_trading: bool = False
    perps_force_mock: bool = False
    perps_position_fraction: float | None = None
    perps_skip_startup_reconcile: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict, repr=False)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    @field_validator("max_leverage", mode="before")
    @classmethod
    def _validate_max_leverage(cls, value: Any) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                "max_leverage_invalid",
                "max_leverage must be a valid integer, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result <= 0:
            raise PydanticCustomError(
                "max_leverage_too_small",
                "max_leverage must be positive, got {value}",
                {"value": result},
            )
        return result

    @field_validator("update_interval", mode="before")
    @classmethod
    def _validate_update_interval(cls, value: Any) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                "update_interval_invalid",
                "update_interval must be a valid integer, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result <= 0:
            raise PydanticCustomError(
                "update_interval_too_small",
                "update_interval must be positive, got {value}",
                {"value": result},
            )
        return result

    @field_validator("max_position_size", mode="before")
    @classmethod
    def _validate_max_position_size(cls, value: Any) -> Decimal:
        try:
            result = Decimal(str(value))
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise PydanticCustomError(
                "max_position_size_invalid",
                "max_position_size must be numeric, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result <= 0:
            raise PydanticCustomError(
                "max_position_size_too_small",
                "max_position_size must be positive, got {value}",
                {"value": str(result)},
            )
        return result

    @field_validator("daily_loss_limit", mode="before")
    @classmethod
    def _validate_daily_loss_limit(cls, value: Any) -> Decimal:
        try:
            result = Decimal(str(value))
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise PydanticCustomError(
                "daily_loss_limit_invalid",
                "daily_loss_limit must be numeric, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result < 0:
            raise PydanticCustomError(
                "daily_loss_limit_negative",
                "daily_loss_limit must be non-negative, got {value}",
                {"value": str(result)},
            )
        return result

    @field_validator("symbols", mode="before")
    @classmethod
    def _validate_symbols(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise PydanticCustomError(
                "symbols_invalid_type",
                "symbols must be a list or tuple, got {type}",
                {"type": type(value).__name__},
            )
        symbols = [str(item).strip().upper() for item in value]
        if not symbols:
            raise PydanticCustomError(
                "symbols_empty",
                "symbols cannot be empty when provided",
                {},
            )
        invalid = [
            f"[{idx}]: {repr(orig)}"
            for idx, (orig, item) in enumerate(zip(value, symbols))
            if not item
        ]
        if invalid:
            raise PydanticCustomError(
                "symbols_invalid_values",
                "symbols must contain only non-empty strings: {invalid}",
                {"invalid": ", ".join(invalid)},
            )
        return symbols

    @field_validator("time_in_force", mode="before")
    @classmethod
    def _validate_time_in_force(cls, value: Any) -> str | None:
        if value is None:
            return None
        tif = str(value).upper()
        supported = {"GTC", "IOC", "FOK"}
        if tif not in supported:
            raise PydanticCustomError(
                "time_in_force_unsupported",
                "time_in_force must be one of {supported}, got {value}",
                {"supported": supported, "value": repr(value)},
            )
        return tif

    @field_validator("account_telemetry_interval", mode="before")
    @classmethod
    def _validate_account_telemetry_interval(cls, value: Any) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                "account_telemetry_interval_invalid",
                "account_telemetry_interval must be a valid integer, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result <= 0:
            raise PydanticCustomError(
                "account_telemetry_interval_too_small",
                "account_telemetry_interval must be positive, got {value}",
                {"value": result},
            )
        return result

    @field_validator("perps_stream_level", mode="before")
    @classmethod
    def _validate_perps_stream_level(cls, value: Any) -> int:
        if value is None:
            return 1
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                "perps_stream_level_invalid",
                "perps_stream_level must be a valid integer, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result < 1:
            raise PydanticCustomError(
                "perps_stream_level_too_small",
                "perps_stream_level must be >= 1, got {value}",
                {"value": result},
            )
        return result

    @field_validator("perps_position_fraction", mode="before")
    @classmethod
    def _validate_perps_position_fraction(cls, value: Any) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                "perps_position_fraction_invalid",
                "perps_position_fraction must be numeric, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if not 0 < result <= 1:
            raise PydanticCustomError(
                "perps_position_fraction_invalid_range",
                "perps_position_fraction must be in (0, 1], got {value}",
                {"value": result},
            )
        return result

    @field_validator("target_leverage", "short_ma", "long_ma", mode="before")
    @classmethod
    def _validate_positive_integers(cls, value: Any, field: Any) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                f"{field.field_name}_invalid",
                f"{field.field_name} must be a valid integer, got {{value}}: {{error}}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result <= 0:
            raise PydanticCustomError(
                f"{field.field_name}_too_small",
                f"{field.field_name} must be positive, got {{value}}",
                {"value": result},
            )
        return result

    @field_validator("trailing_stop_pct", mode="before")
    @classmethod
    def _validate_trailing_stop_pct(cls, value: Any) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                "trailing_stop_pct_invalid",
                "trailing_stop_pct must be numeric, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result < 0:
            raise PydanticCustomError(
                "trailing_stop_pct_negative",
                "trailing_stop_pct must be non-negative, got {value}",
                {"value": result},
            )
        return result

    @model_validator(mode="after")
    def _apply_defaults_and_normalization(self) -> BotConfig:
        metadata = dict(self.metadata) if isinstance(self.metadata, dict) else {}

        settings = metadata.get("_runtime_settings")
        if not isinstance(settings, RuntimeSettings):
            settings = load_runtime_settings()
            metadata["_runtime_settings"] = settings
            object.__setattr__(self, "metadata", metadata)

        submitted_symbols = list(self.symbols) if self.symbols is not None else None
        last_normalized = metadata.get("normalized_symbols")
        last_normalized_list = list(last_normalized) if isinstance(last_normalized, list) else None
        requested_symbols = metadata.get("requested_symbols")
        requested_list = list(requested_symbols) if isinstance(requested_symbols, list) else None

        override_payload = metadata.get("symbol_normalization_overrides", {})
        override_quote: str | None = None
        override_derivatives: bool | None = None
        if isinstance(override_payload, dict):
            raw_quote = override_payload.get("quote")
            if raw_quote is not None:
                override_quote = str(raw_quote).upper()
            if "allow_derivatives" in override_payload:
                override_derivatives = bool(override_payload["allow_derivatives"])

        if self.symbols is None:
            object.__setattr__(self, "symbols", list(DEFAULT_SPOT_SYMBOLS))
            submitted_symbols = list(DEFAULT_SPOT_SYMBOLS)

        if submitted_symbols is not None:
            if last_normalized_list is not None and submitted_symbols != last_normalized_list:
                requested_list = list(submitted_symbols)
                metadata["requested_symbols"] = list(submitted_symbols)
            elif requested_list is None:
                requested_list = list(submitted_symbols)
                metadata["requested_symbols"] = list(submitted_symbols)

        input_symbols = requested_list if requested_list is not None else submitted_symbols

        metadata.setdefault("default_quote", settings.coinbase_default_quote)

        quote_currency = (
            override_quote
            or metadata.get("default_quote")
            or settings.coinbase_default_quote
            or "USD"
        )
        quote_currency = str(quote_currency).upper()
        if override_derivatives is None:
            allow_derivatives = _resolve_derivatives_enabled(self.profile)
        else:
            allow_derivatives = override_derivatives

        normalized, logs = normalize_symbol_list(
            input_symbols,
            allow_derivatives=allow_derivatives,
            quote=quote_currency,
        )
        object.__setattr__(self, "symbols", normalized)
        object.__setattr__(self, "derivatives_enabled", bool(allow_derivatives))

        metadata.setdefault("default_quote", quote_currency)
        metadata["symbol_normalization_logs"] = [
            {"level": record.level, "message": record.message, "args": list(record.args)}
            for record in logs
        ]
        metadata["normalized_symbols"] = list(normalized)

        if isinstance(override_payload, dict):
            existing_quote = override_payload.get("quote")
            existing_allow = override_payload.get("allow_derivatives")
        else:
            existing_quote = None
            existing_allow = None

        if (
            override_quote is not None
            or override_derivatives is not None
            or existing_quote is not None
            or existing_allow is not None
        ):
            metadata["symbol_normalization_overrides"] = {
                "quote": override_quote if override_quote is not None else existing_quote,
                "allow_derivatives": (
                    override_derivatives if override_derivatives is not None else existing_allow
                ),
            }
        elif "symbol_normalization_overrides" in metadata:
            metadata.pop("symbol_normalization_overrides", None)

        object.__setattr__(self, "metadata", metadata)
        return self

    @model_validator(mode="after")
    def _validate_profile_constraints(self) -> BotConfig:
        errors: list[str] = []

        if self.profile == Profile.CANARY:
            if not self.reduce_only_mode:
                errors.append(
                    "[canary_reduce_only_required] Canary profile must have reduce_only_mode=True"
                )
            if self.max_leverage > 1:
                errors.append(
                    f"[canary_max_leverage_too_high] Canary profile max_leverage cannot exceed 1, got {self.max_leverage}"
                )
            if self.time_in_force not in ("IOC",):
                errors.append(
                    f"[canary_time_in_force_invalid] Canary profile must use IOC time_in_force, got {self.time_in_force}"
                )

        elif self.profile == Profile.SPOT:
            if self.enable_shorts:
                errors.append("[spot_shorts_not_allowed] Spot profile cannot enable shorts")
            if self.max_leverage > 1:
                errors.append(
                    f"[spot_leverage_too_high] Spot profile max_leverage cannot exceed 1, got {self.max_leverage}"
                )

        if self.short_ma >= self.long_ma:
            errors.append(
                f"[ma_periods_invalid] short_ma ({self.short_ma}) must be < long_ma ({self.long_ma})"
            )

        if errors:
            raise ValueError("; ".join(errors))
        return self

    @classmethod
    def from_profile(
        cls, profile: str, *, settings: RuntimeSettings | None = None, **overrides: Any
    ) -> BotConfig:
        manager = ConfigManager(
            profile=profile,
            overrides=overrides,
            config_cls=cls,
            settings=settings,
        )
        return manager.build()


class ConfigValidationError(Exception):
    """Raised when configuration values fail validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = "; ".join(errors) if errors else "Invalid configuration"
        super().__init__(message)


class ConfigManager:
    """Centralizes profile configuration loading, overrides, and validation."""

    ENVIRONMENT_KEYS = (
        "COINBASE_ENABLE_DERIVATIVES",
        "COINBASE_DEFAULT_QUOTE",
        "ORDER_PREVIEW_ENABLED",
        "PERPS_ENABLE_STREAMING",
        "PERPS_FORCE_MOCK",
        "PERPS_PAPER",
        "PERPS_POSITION_FRACTION",
        "PERPS_SKIP_RECONCILE",
        "PERPS_STREAM_LEVEL",
        "SLIPPAGE_MULTIPLIERS",
        "SPOT_FORCE_LIVE",
    )

    CANARY_LOCKED_KEYS = {"reduce_only_mode", "max_leverage", "time_in_force"}

    def __init__(
        self,
        profile: str | Profile,
        overrides: dict[str, Any] | None = None,
        config_cls: type[BotConfig] = BotConfig,
        auto_build: bool = True,
        settings: RuntimeSettings | None = None,
    ) -> None:
        self.profile = profile if isinstance(profile, Profile) else Profile(profile)
        self.overrides = dict(overrides or {})
        self._config_cls = config_cls
        self._config: BotConfig | None = None
        self._last_snapshot: dict[str, Any] | None = None
        self._settings_locked = settings is not None
        self._settings: RuntimeSettings = settings or load_runtime_settings()
        if auto_build:
            self.build()

    # ----- Public API -------------------------------------------------
    def _create_config(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BotConfig:
        meta = dict(metadata or {})
        meta.setdefault("_runtime_settings", self._settings)
        return cast(BotConfig, self._config_cls(metadata=meta, **kwargs))

    def build(self) -> BotConfig:
        """Construct a validated configuration instance."""

        self._refresh_settings_if_needed()
        config = self._build_profile_config()
        config = self._apply_overrides(config)
        config = self._post_process(config)
        self._validate(config)

        snapshot = self._capture_snapshot()
        config.metadata["profile"] = self.profile.value
        config.metadata["overrides"] = deepcopy(self.overrides)
        config.metadata["config_snapshot"] = snapshot

        self._config = config
        self._last_snapshot = snapshot
        return config

    def has_changes(self) -> bool:
        if self._last_snapshot is None:
            return True
        return self._capture_snapshot() != self._last_snapshot

    def refresh_if_changed(self) -> BotConfig | None:
        """Rebuild configuration when underlying inputs change."""

        if not self.has_changes():
            return None
        return self.build()

    def get_config(self) -> BotConfig | None:
        return self._config

    @classmethod
    def from_config(
        cls, config: BotConfig, settings: RuntimeSettings | None = None
    ) -> ConfigManager:
        """Recreate a manager from an existing configuration instance."""

        overrides = deepcopy(config.metadata.get("overrides", {}))
        snapshot = deepcopy(config.metadata.get("config_snapshot"))
        manager = cls(
            profile=config.profile,
            overrides=overrides,
            config_cls=type(config),
            auto_build=False,
            settings=settings,
        )
        manager._config = config
        if snapshot is not None:
            manager._last_snapshot = snapshot
        else:
            manager._last_snapshot = manager._capture_snapshot()
        manager._validate(config)
        return manager

    # ----- Internal helpers -------------------------------------------
    def _build_profile_config(self) -> BotConfig:
        if self.profile == Profile.CANARY:
            return self._build_canary_config()
        if self.profile == Profile.DEV:
            return self._create_config(
                profile=self.profile,
                mock_broker=True,
                mock_fills=True,
                max_position_size=Decimal("10000"),
                dry_run=True,
            )
        if self.profile == Profile.DEMO:
            return self._create_config(
                profile=self.profile,
                max_position_size=Decimal("100"),
                max_leverage=1,
                enable_shorts=False,
            )
        if self.profile == Profile.SPOT:
            return self._create_config(
                profile=self.profile,
                max_position_size=Decimal("50000"),
                max_leverage=1,
                enable_shorts=False,
                mock_broker=False,
                mock_fills=False,
            )
        # Default to production profile (perps capable)
        return self._create_config(
            profile=self.profile,
            max_position_size=Decimal("50000"),
            max_leverage=3,
            enable_shorts=True,
        )

    def _build_canary_config(self) -> BotConfig:
        symbols = ["BTC-USD"]
        max_leverage = 1
        reduce_only = True
        update_interval = 5
        trading_window_start: time | None = None
        trading_window_end: time | None = None
        trading_days: list[str] | None = None
        daily_loss_limit = Decimal("10")
        time_in_force = "IOC"

        profile_path = path_registry.PROJECT_ROOT / "config" / "profiles" / "canary.yaml"
        if profile_path.exists():
            try:
                import yaml  # type: ignore

                with profile_path.open("r") as handle:
                    payload = yaml.safe_load(handle) or {}
                symbols = payload.get("trading", {}).get("symbols", symbols)
                reduce_only = payload.get("trading", {}).get(
                    "mode"
                ) == "reduce_only" or payload.get("features", {}).get("reduce_only_mode", True)
                max_leverage = int(
                    payload.get("risk_management", {}).get("max_leverage", max_leverage)
                )
                update_interval = int(
                    payload.get("monitoring", {})
                    .get("metrics", {})
                    .get("interval_seconds", update_interval)
                )
                session = payload.get("session", {})
                start_str = session.get("start_time")
                end_str = session.get("end_time")
                trading_window_start = time.fromisoformat(start_str) if start_str else None
                trading_window_end = time.fromisoformat(end_str) if end_str else None
                trading_days = session.get(
                    "days", ["monday", "tuesday", "wednesday", "thursday", "friday"]
                )
                daily_loss_limit = Decimal(
                    str(
                        payload.get("risk_management", {}).get("daily_loss_limit", daily_loss_limit)
                    )
                )
                time_in_force = payload.get("order_policy", {}).get("time_in_force", time_in_force)
            except Exception as exc:  # pragma: no cover - defensive logging
                import logging

                logger = logging.getLogger(__name__)
                logger.debug("Failed to load canary profile YAML: %s", exc, exc_info=True)

        return self._create_config(
            profile=self.profile,
            symbols=symbols,
            reduce_only_mode=reduce_only,
            max_leverage=max_leverage,
            update_interval=update_interval,
            dry_run=False,
            max_position_size=Decimal("500"),
            trading_window_start=trading_window_start,
            trading_window_end=trading_window_end,
            trading_days=trading_days,
            daily_loss_limit=daily_loss_limit,
            time_in_force=time_in_force,
        )

    def _apply_overrides(self, config: BotConfig) -> BotConfig:
        if not self.overrides:
            return config

        errors: list[str] = []
        for key, value in self.overrides.items():
            if value is None:
                continue
            if self.profile == Profile.CANARY and key in self.CANARY_LOCKED_KEYS:
                continue
            if not hasattr(config, key):
                errors.append(f"Unknown configuration override '{key}'")
                continue
            try:
                setattr(config, key, value)
            except ValidationError as exc:
                errors.extend(self._format_validation_errors(exc))
            except (TypeError, ValueError) as exc:
                errors.append(f"{key}: {exc}")

        if errors:
            raise ConfigValidationError(errors)
        return config

    def _post_process(self, config: BotConfig) -> BotConfig:
        if self.profile == Profile.CANARY:
            config.reduce_only_mode = True
            try:
                config.max_leverage = min(int(getattr(config, "max_leverage", 1)), 1)
            except Exception:
                config.max_leverage = 1
            config.time_in_force = "IOC"
        elif self.profile == Profile.SPOT:
            config.enable_shorts = False
            config.max_leverage = 1
            config.reduce_only_mode = False
            if not self._settings.spot_force_live:
                logger.info("Spot profile detected; falling back to mock broker for safety")
                config.mock_broker = True

        config = self._apply_runtime_toggles(config)

        if (
            "enable_order_preview" not in self.overrides
            or self.overrides.get("enable_order_preview") is None
        ):
            preview_flag = self._settings.order_preview_enabled
            if preview_flag is not None:
                config.enable_order_preview = preview_flag

        return config

    def _validate(self, config: BotConfig) -> None:
        """Validate configuration using pydantic schema."""
        try:
            self._config_cls.model_validate(config.model_dump())
        except ValidationError as exc:
            raise ConfigValidationError(self._format_validation_errors(exc))

    def validate_config_dict(self, config_dict: dict[str, Any]) -> ConfigValidationResult:
        """Validate a configuration dictionary using pydantic schema.

        Returns detailed validation results instead of raising exceptions.
        """
        try:
            self._config_cls.model_validate(config_dict)
            return ConfigValidationResult(is_valid=True)
        except ValidationError as exc:
            errors = self._format_validation_errors(exc)
            return ConfigValidationResult(is_valid=False, errors=errors, warnings=[])

    def _apply_runtime_toggles(self, config: BotConfig) -> BotConfig:
        settings = self._settings

        config.perps_enable_streaming = settings.perps_enable_streaming
        config.perps_stream_level = settings.perps_stream_level

        config.perps_paper_trading = settings.perps_paper_trading
        config.perps_force_mock = settings.perps_force_mock
        config.perps_skip_startup_reconcile = settings.perps_skip_startup_reconcile
        config.perps_position_fraction = settings.perps_position_fraction

        return config

    def _capture_snapshot(self) -> dict[str, Any]:
        snapshot_settings = self._settings if self._settings_locked else load_runtime_settings()
        env_snapshot = snapshot_settings.snapshot_env(self.ENVIRONMENT_KEYS)
        env_snapshot["profile"] = self.profile.value

        overrides_snapshot = {
            key: self._normalize_snapshot_value(value)
            for key, value in sorted(self.overrides.items())
        }

        file_snapshot: dict[str, Any] = {}
        for path in self._files_to_watch():
            try:
                file_snapshot[str(path)] = path.stat().st_mtime if path.exists() else None
            except Exception:
                file_snapshot[str(path)] = None

        return {
            "env": env_snapshot,
            "files": file_snapshot,
            "overrides": overrides_snapshot,
        }

    def _files_to_watch(self) -> list[Path]:
        if self.profile == Profile.CANARY:
            return [path_registry.PROJECT_ROOT / "config" / "profiles" / "canary.yaml"]
        return []

    @staticmethod
    def _normalize_snapshot_value(value: Any) -> Any:
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [ConfigManager._normalize_snapshot_value(v) for v in value]
        if isinstance(value, dict):
            return {k: ConfigManager._normalize_snapshot_value(v) for k, v in value.items()}
        return value

    @staticmethod
    def _format_validation_errors(exc: ValidationError) -> list[str]:
        errors: list[str] = []
        for error in exc.errors():
            loc = error.get("loc", ())
            field_path = ".".join(str(item) for item in loc)
            message = error.get("msg", "")
            if field_path:
                errors.append(f"{field_path}: {message}")
            else:
                errors.append(message or str(exc))
        if not errors:
            errors.append(str(exc))
        return errors

    def _refresh_settings_if_needed(self) -> None:
        if not self._settings_locked:
            self._settings = load_runtime_settings()


class ConfigValidationResult(BaseModel):
    """Result of configuration validation."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if validation failed."""
        return bool(self.errors)

    @property
    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return bool(self.warnings)

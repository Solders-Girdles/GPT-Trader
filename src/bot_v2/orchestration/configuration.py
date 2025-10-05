"""Configuration primitives for GPT-Trader orchestration."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, cast

from bot_v2.orchestration.symbols import (
    derivatives_enabled as _resolve_derivatives_enabled,
)
from bot_v2.orchestration.symbols import (
    normalize_symbols,
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


class Profile(Enum):
    """Configuration profiles."""

    DEV = "dev"
    DEMO = "demo"
    STAGING = "staging"
    PROD = "prod"
    CANARY = "canary"
    SPOT = "spot"


@dataclass
class BotConfig:
    """Bot configuration."""

    profile: Profile
    dry_run: bool = False
    symbols: Sequence[str] | None = None
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
    max_trade_value: Decimal = Decimal("0")
    symbol_position_caps: dict[str, Decimal] = field(default_factory=dict)
    time_in_force: str = "GTC"
    perps_enable_streaming: bool = False
    perps_stream_level: int = 1
    perps_paper_trading: bool = False
    perps_force_mock: bool = False
    perps_position_fraction: float | None = None
    perps_skip_startup_reconcile: bool = False
    streaming_rest_poll_interval: float = 5.0
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.symbols is None:
            # Default to high-volume spot pairs; derivatives mode can override via CLI/env.
            self.symbols = list(DEFAULT_SPOT_SYMBOLS)
        if not isinstance(self.metadata, dict):
            self.metadata = {}

    @classmethod
    def from_profile(cls, profile: str, **overrides: Any) -> BotConfig:
        manager = ConfigManager(profile=profile, overrides=overrides, config_cls=cls)
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
        "PERPS_STREAMING_REST_INTERVAL",
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
    ) -> None:
        self.profile = profile if isinstance(profile, Profile) else Profile(profile)
        self.overrides = dict(overrides or {})
        self._config_cls = config_cls
        self._config: BotConfig | None = None
        self._last_snapshot: dict[str, Any] | None = None
        if auto_build:
            self.build()

    # ----- Public API -------------------------------------------------
    def build(self) -> BotConfig:
        """Construct a validated configuration instance."""

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
    def from_config(cls, config: BotConfig) -> ConfigManager:
        """Recreate a manager from an existing configuration instance."""

        overrides = deepcopy(config.metadata.get("overrides", {}))
        snapshot = deepcopy(config.metadata.get("config_snapshot"))
        manager = cls(
            profile=config.profile, overrides=overrides, config_cls=type(config), auto_build=False
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
            return cast(
                BotConfig,
                self._config_cls(
                    profile=self.profile,
                    mock_broker=True,
                    mock_fills=True,
                    max_position_size=Decimal("10000"),
                    dry_run=True,
                ),
            )
        if self.profile == Profile.DEMO:
            return cast(
                BotConfig,
                self._config_cls(
                    profile=self.profile,
                    max_position_size=Decimal("100"),
                    max_leverage=1,
                    enable_shorts=False,
                ),
            )
        if self.profile == Profile.SPOT:
            return cast(
                BotConfig,
                self._config_cls(
                    profile=self.profile,
                    max_position_size=Decimal("50000"),
                    max_leverage=1,
                    enable_shorts=False,
                    mock_broker=False,
                    mock_fills=False,
                ),
            )
        # Default to production profile (perps capable)
        return cast(
            BotConfig,
            self._config_cls(
                profile=self.profile,
                max_position_size=Decimal("50000"),
                max_leverage=3,
                enable_shorts=True,
            ),
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

        profile_path = Path("config/profiles/canary.yaml")
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

        return cast(
            BotConfig,
            self._config_cls(
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
            ),
        )

    def _apply_overrides(self, config: BotConfig) -> BotConfig:
        for key, value in self.overrides.items():
            if value is None:
                continue
            if self.profile == Profile.CANARY and key in self.CANARY_LOCKED_KEYS:
                continue
            if hasattr(config, key):
                setattr(config, key, value)
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
            if os.getenv("SPOT_FORCE_LIVE", "").lower() not in {"1", "true"}:
                logger.info("Spot profile detected; falling back to mock broker for safety")
                config.mock_broker = True

        config = self._normalize_symbols(config)
        config = self._apply_runtime_toggles(config)

        if (
            "enable_order_preview" not in self.overrides
            or self.overrides.get("enable_order_preview") is None
        ):
            preview_env = os.getenv("ORDER_PREVIEW_ENABLED")
            if preview_env is not None:
                config.enable_order_preview = preview_env.lower() in {"1", "true", "yes"}

        return config

    def _validate(self, config: BotConfig) -> None:
        errors: list[str] = []

        if not isinstance(config.profile, Profile):
            errors.append("profile must be a Profile enum value")

        symbols = list(config.symbols or [])
        if not symbols or not all(isinstance(sym, str) and sym.strip() for sym in symbols):
            errors.append("symbols must contain at least one non-empty string")

        if config.update_interval <= 0:
            errors.append("update_interval must be > 0")

        try:
            if Decimal(str(config.max_position_size)) <= Decimal("0"):
                errors.append("max_position_size must be positive")
        except Exception:
            errors.append("max_position_size must be numeric")

        try:
            if int(config.max_leverage) <= 0:
                errors.append("max_leverage must be positive")
        except Exception:
            errors.append("max_leverage must be an integer")

        try:
            if float(config.streaming_rest_poll_interval) <= 0:
                errors.append("streaming_rest_poll_interval must be positive")
        except Exception:
            errors.append("streaming_rest_poll_interval must be numeric")

        tif = (config.time_in_force or "").upper()
        if tif and tif not in {"GTC", "IOC", "FOK"}:
            errors.append(f"Unsupported time_in_force '{config.time_in_force}'")

        if errors:
            raise ConfigValidationError(errors)

    def _normalize_symbols(self, config: BotConfig) -> BotConfig:
        try:
            normalized, allow_derivatives = normalize_symbols(config.profile, config.symbols)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to normalize symbol list: %s", exc, exc_info=True)
            normalized = list(config.symbols or [])
            allow_derivatives = _resolve_derivatives_enabled(config.profile)

        config.symbols = tuple(normalized)
        config.derivatives_enabled = bool(allow_derivatives)
        return config

    def _apply_runtime_toggles(self, config: BotConfig) -> BotConfig:
        def _flag(name: str) -> bool:
            return os.getenv(name, "").lower() in {"1", "true", "yes"}

        config.perps_enable_streaming = _flag("PERPS_ENABLE_STREAMING")

        stream_level_raw = os.getenv("PERPS_STREAM_LEVEL")
        try:
            config.perps_stream_level = int(stream_level_raw) if stream_level_raw else 1
        except (TypeError, ValueError):
            logger.warning("Invalid PERPS_STREAM_LEVEL=%s; defaulting to 1", stream_level_raw)
            config.perps_stream_level = 1

        config.perps_paper_trading = _flag("PERPS_PAPER")
        config.perps_force_mock = _flag("PERPS_FORCE_MOCK")
        config.perps_skip_startup_reconcile = _flag("PERPS_SKIP_RECONCILE")

        position_fraction_raw = os.getenv("PERPS_POSITION_FRACTION")
        if position_fraction_raw:
            try:
                config.perps_position_fraction = float(position_fraction_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid PERPS_POSITION_FRACTION=%s; ignoring override",
                    position_fraction_raw,
                )
                config.perps_position_fraction = None
        else:
            config.perps_position_fraction = None

        rest_interval_raw = os.getenv("PERPS_STREAMING_REST_INTERVAL")
        if rest_interval_raw:
            try:
                interval = float(rest_interval_raw)
                if interval <= 0:
                    raise ValueError("interval must be positive")
                config.streaming_rest_poll_interval = interval
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid PERPS_STREAMING_REST_INTERVAL=%s; ignoring override",
                    rest_interval_raw,
                )

        return config

    def _capture_snapshot(self) -> dict[str, Any]:
        env_snapshot = {key: os.getenv(key) for key in self.ENVIRONMENT_KEYS}
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
            return [Path("config/profiles/canary.yaml")]
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

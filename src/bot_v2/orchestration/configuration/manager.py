"""Config manager for orchestrating profile creation, overrides, and validation."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from bot_v2.config import path_registry
from bot_v2.config.types import Profile
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.utilities.logging_patterns import get_logger

from .core import BotConfig
from .profiles import build_profile_config
from .validation import (
    ConfigValidationError,
    ConfigValidationResult,
    format_validation_errors,
)

logger = get_logger(__name__, component="config_manager")


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
        config = cast(BotConfig, self._config_cls(metadata=meta, **kwargs))
        config.state.runtime_settings = self._settings
        return config

    def build(self) -> BotConfig:
        """Construct a validated configuration instance."""

        self._refresh_settings_if_needed()
        config = self._build_profile_config()
        config = self._apply_overrides(config)
        config = self._post_process(config)
        self._validate(config)

        snapshot = self._capture_snapshot()
        config.state.profile_value = self.profile.value
        config.state.overrides_snapshot = deepcopy(self.overrides)
        config.state.config_snapshot = snapshot

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

    def replace_config(self, config: BotConfig) -> None:
        """Replace the managed configuration with a new instance."""

        self._config = config
        snapshot = self._capture_snapshot()
        self._last_snapshot = snapshot
        config.state.config_snapshot = snapshot
        config.state.overrides_snapshot = deepcopy(self.overrides)
        config.state.profile_value = self.profile.value

    @classmethod
    def from_config(
        cls, config: BotConfig, settings: RuntimeSettings | None = None
    ) -> ConfigManager:
        """Recreate a manager from an existing configuration instance."""

        overrides = deepcopy(config.state.overrides_snapshot)
        snapshot = deepcopy(config.state.config_snapshot)
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
        return build_profile_config(self.profile, self._create_config)

    def _apply_overrides(self, config: BotConfig) -> BotConfig:
        if not self.overrides:
            return config

        errors: list[str] = []
        updated = config
        for key, value in self.overrides.items():
            if value is None:
                continue
            if self.profile == Profile.CANARY and key in self.CANARY_LOCKED_KEYS:
                continue
            if not hasattr(config, key):
                errors.append(f"Unknown configuration override '{key}'")
                continue
            try:
                updated = updated.with_overrides(**{key: value})
            except ValidationError as exc:
                errors.extend(format_validation_errors(exc))
            except (TypeError, ValueError) as exc:
                errors.append(f"{key}: {exc}")

        if errors:
            raise ConfigValidationError(errors)
        return updated

    def _post_process(self, config: BotConfig) -> BotConfig:
        profile_updates: dict[str, Any] = {}
        if self.profile == Profile.CANARY:
            try:
                leverage_value = int(getattr(config, "max_leverage", 1))
            except Exception:
                leverage_value = 1
            profile_updates.update(
                reduce_only_mode=True,
                max_leverage=min(leverage_value, 1),
                time_in_force="IOC",
            )
        elif self.profile == Profile.SPOT:
            profile_updates.update(
                enable_shorts=False,
                max_leverage=1,
                reduce_only_mode=False,
            )
            if not self._settings.spot_force_live:
                logger.info(
                    "Spot profile detected; falling back to mock broker",
                    operation="config_manager",
                    stage="post_process",
                )
                profile_updates["mock_broker"] = True

        if profile_updates:
            config = config.with_overrides(**profile_updates)

        config = self._apply_runtime_toggles(config)

        if (
            "enable_order_preview" not in self.overrides
            or self.overrides.get("enable_order_preview") is None
        ):
            preview_flag = self._settings.order_preview_enabled
            if preview_flag is not None:
                config = config.with_overrides(enable_order_preview=preview_flag)

        return config

    def _validate(self, config: BotConfig) -> None:
        """Validate configuration using pydantic schema."""
        try:
            self._config_cls.model_validate(config.model_dump())
        except ValidationError as exc:
            raise ConfigValidationError(format_validation_errors(exc))

    def validate_config_dict(self, config_dict: dict[str, Any]) -> ConfigValidationResult:
        """Validate a configuration dictionary using pydantic schema.

        Returns detailed validation results instead of raising exceptions.
        """
        try:
            self._config_cls.model_validate(config_dict)
            return ConfigValidationResult(is_valid=True)
        except ValidationError as exc:
            errors = format_validation_errors(exc)
            return ConfigValidationResult(is_valid=False, errors=errors, warnings=[])

    def _apply_runtime_toggles(self, config: BotConfig) -> BotConfig:
        settings = self._settings

        return config.with_overrides(
            perps_enable_streaming=settings.perps_enable_streaming,
            perps_stream_level=settings.perps_stream_level,
            perps_paper_trading=settings.perps_paper_trading,
            perps_force_mock=settings.perps_force_mock,
            us_futures_enabled=settings.coinbase_us_futures_enabled,
            intx_perpetuals_enabled=settings.coinbase_intx_perpetuals_enabled,
            derivatives_type=settings.coinbase_derivatives_type,
            perps_skip_startup_reconcile=settings.perps_skip_startup_reconcile,
            perps_position_fraction=settings.perps_position_fraction,
        )

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

    def _refresh_settings_if_needed(self) -> None:
        if not self._settings_locked:
            self._settings = load_runtime_settings()


__all__ = ["ConfigManager"]

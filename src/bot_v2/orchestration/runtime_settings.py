from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from bot_v2.config.path_registry import RUNTIME_DATA_DIR
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.parsing import interpret_tristate_bool

logger = get_logger(__name__, component="runtime_settings")


def _normalize_bool(value: str | None, *, field_name: str | None = None) -> bool | None:
    if value is None:
        return None
    interpreted = interpret_tristate_bool(value)
    if interpreted is not None:
        return interpreted
    stripped = value.strip()
    if stripped and field_name:
        logger.warning(
            "Invalid %s=%s; ignoring override",
            field_name,
            value,
            operation="runtime_setting_parse",
            status="invalid",
            field=field_name,
            raw=value,
        )
    return None


def _safe_int(value: str | None, *, fallback: int, field_name: str) -> int:
    if value is None or not value.strip():
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%s; defaulting to %s",
            field_name,
            value,
            fallback,
            operation="runtime_setting_parse",
            status="fallback",
            field=field_name,
            raw=value,
            fallback_value=fallback,
        )
        return fallback
    return parsed


def _safe_float(value: str | None, *, field_name: str) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%s; ignoring override",
            field_name,
            value,
            operation="runtime_setting_parse",
            status="invalid",
            field=field_name,
            raw=value,
        )
        return None


@dataclass
class RuntimeSettings:
    """Normalized snapshot of runtime-affecting environment toggles."""

    raw_env: dict[str, str] = field(default_factory=dict)
    runtime_root: Path = Path(".")
    event_store_root_override: Path | None = None
    coinbase_default_quote: str = "USD"
    coinbase_default_quote_overridden: bool = False
    coinbase_enable_derivatives: bool = False
    coinbase_enable_derivatives_overridden: bool = False
    coinbase_us_futures_enabled: bool = False
    coinbase_intx_perpetuals_enabled: bool = False
    coinbase_derivatives_type: str = "intx_perps"
    perps_enable_streaming: bool = False
    perps_stream_level: int = 1
    perps_paper_trading: bool = False
    perps_force_mock: bool = False
    perps_skip_startup_reconcile: bool = False
    perps_position_fraction: float | None = None
    order_preview_enabled: bool | None = None
    spot_force_live: bool = False
    broker_hint: str | None = None
    coinbase_sandbox_enabled: bool = False
    coinbase_api_mode: str = "advanced"
    risk_config_path: Path | None = None
    coinbase_intx_portfolio_uuid: str | None = None
    environment: str | None = None
    trading_enabled: bool | None = None
    max_position_size: float | None = None
    max_daily_loss: float | None = None
    circuit_breaker_enabled: bool | None = None
    monitoring_enabled: bool | None = None

    def snapshot_env(
        self, keys: Mapping[str, object] | list[str] | tuple[str, ...]
    ) -> dict[str, str | None]:
        if isinstance(keys, Mapping):
            iterable: list[str] = list(str(key) for key in keys.keys())
        else:
            iterable = list(keys)
        return {key: self.raw_env.get(key) for key in iterable}


def load_runtime_settings(env: Mapping[str, str] | None = None) -> RuntimeSettings:
    """Capture relevant runtime settings from environment variables."""

    env_map = dict(env or os.environ)

    runtime_root = Path(env_map.get("GPT_TRADER_RUNTIME_ROOT") or RUNTIME_DATA_DIR)

    event_store_raw = env_map.get("EVENT_STORE_ROOT")
    event_store_root = Path(event_store_raw) if event_store_raw else None

    default_quote_raw = env_map.get("COINBASE_DEFAULT_QUOTE")
    default_quote = (default_quote_raw or "USD").upper()

    derivatives_flag_raw = env_map.get("COINBASE_ENABLE_DERIVATIVES")
    derivatives_bool = _normalize_bool(
        derivatives_flag_raw, field_name="COINBASE_ENABLE_DERIVATIVES"
    )
    coinbase_enable_derivatives = bool(derivatives_bool)

    us_futures_flag_raw = env_map.get("COINBASE_US_FUTURES_ENABLED")
    coinbase_us_futures_enabled = bool(
        _normalize_bool(us_futures_flag_raw, field_name="COINBASE_US_FUTURES_ENABLED")
    )

    intx_perps_flag_raw = env_map.get("COINBASE_INTX_PERPETUALS_ENABLED")
    coinbase_intx_perpetuals_enabled = bool(
        _normalize_bool(intx_perps_flag_raw, field_name="COINBASE_INTX_PERPETUALS_ENABLED")
    )

    derivatives_type_raw = env_map.get("COINBASE_DERIVATIVES_TYPE")
    coinbase_derivatives_type = (derivatives_type_raw or "intx_perps").lower()

    perps_enable_streaming = bool(
        _normalize_bool(env_map.get("PERPS_ENABLE_STREAMING"), field_name="PERPS_ENABLE_STREAMING")
    )
    perps_stream_level = max(
        1,
        _safe_int(env_map.get("PERPS_STREAM_LEVEL"), fallback=1, field_name="PERPS_STREAM_LEVEL"),
    )
    perps_paper = bool(_normalize_bool(env_map.get("PERPS_PAPER"), field_name="PERPS_PAPER"))
    perps_force_mock = bool(
        _normalize_bool(env_map.get("PERPS_FORCE_MOCK"), field_name="PERPS_FORCE_MOCK")
    )
    perps_skip_reconcile = bool(
        _normalize_bool(env_map.get("PERPS_SKIP_RECONCILE"), field_name="PERPS_SKIP_RECONCILE")
    )
    perps_fraction = _safe_float(
        env_map.get("PERPS_POSITION_FRACTION"), field_name="PERPS_POSITION_FRACTION"
    )

    order_preview_raw = _normalize_bool(
        env_map.get("ORDER_PREVIEW_ENABLED"), field_name="ORDER_PREVIEW_ENABLED"
    )
    spot_force_live = bool(
        _normalize_bool(env_map.get("SPOT_FORCE_LIVE"), field_name="SPOT_FORCE_LIVE")
    )

    broker_hint = env_map.get("BROKER")
    coinbase_sandbox = bool(
        _normalize_bool(env_map.get("COINBASE_SANDBOX"), field_name="COINBASE_SANDBOX")
    )
    coinbase_api_mode = (env_map.get("COINBASE_API_MODE") or "advanced").lower()
    coinbase_intx_portfolio_uuid = env_map.get("COINBASE_INTX_PORTFOLIO_UUID")

    risk_config_raw = env_map.get("RISK_CONFIG_PATH")
    risk_config_path = Path(risk_config_raw) if risk_config_raw else None

    return RuntimeSettings(
        raw_env=env_map,
        runtime_root=runtime_root,
        event_store_root_override=event_store_root,
        coinbase_default_quote=default_quote,
        coinbase_default_quote_overridden=default_quote_raw is not None,
        coinbase_enable_derivatives=coinbase_enable_derivatives,
        coinbase_enable_derivatives_overridden=derivatives_flag_raw is not None,
        coinbase_us_futures_enabled=coinbase_us_futures_enabled,
        coinbase_intx_perpetuals_enabled=coinbase_intx_perpetuals_enabled,
        coinbase_derivatives_type=coinbase_derivatives_type,
        perps_enable_streaming=perps_enable_streaming,
        perps_stream_level=perps_stream_level,
        perps_paper_trading=perps_paper,
        perps_force_mock=perps_force_mock,
        perps_skip_startup_reconcile=perps_skip_reconcile,
        perps_position_fraction=perps_fraction,
        order_preview_enabled=order_preview_raw,
        spot_force_live=spot_force_live,
        broker_hint=broker_hint.lower() if isinstance(broker_hint, str) else None,
        coinbase_sandbox_enabled=coinbase_sandbox,
        coinbase_api_mode=coinbase_api_mode,
        risk_config_path=risk_config_path,
        coinbase_intx_portfolio_uuid=coinbase_intx_portfolio_uuid,
        environment=env_map.get("ENVIRONMENT"),
        trading_enabled=None,
        max_position_size=None,
        max_daily_loss=None,
        circuit_breaker_enabled=None,
        monitoring_enabled=None,
    )


_SNAPSHOT_CACHE: RuntimeSettings | None = None
_OVERRIDE_SNAPSHOT: RuntimeSettings | None = None


def set_runtime_settings_override(snapshot: RuntimeSettings | None) -> None:
    """Force downstream callers to use the provided snapshot (primarily for tests)."""

    global _OVERRIDE_SNAPSHOT, _SNAPSHOT_CACHE
    _OVERRIDE_SNAPSHOT = snapshot
    if snapshot is not None:
        _SNAPSHOT_CACHE = snapshot


def clear_runtime_settings_cache() -> None:
    """Clear the cached snapshot so the next lookup reloads from the environment."""

    global _SNAPSHOT_CACHE
    _SNAPSHOT_CACHE = None


def get_runtime_settings(
    *,
    force_refresh: bool = False,
    env: Mapping[str, str] | None = None,
) -> RuntimeSettings:
    """Return a cached runtime settings snapshot or create one if needed."""

    if env is not None:
        # Explicit environment requested – bypass caches/overrides
        return load_runtime_settings(env)

    if _OVERRIDE_SNAPSHOT is not None and not force_refresh:
        return _OVERRIDE_SNAPSHOT

    global _SNAPSHOT_CACHE
    if force_refresh or _SNAPSHOT_CACHE is None:
        _SNAPSHOT_CACHE = load_runtime_settings()
    return _SNAPSHOT_CACHE


class RuntimeSettingsProvider:
    """Thin wrapper around module-level runtime settings helpers."""

    def __init__(
        self,
        loader: Callable[[Mapping[str, str] | None], RuntimeSettings] = load_runtime_settings,
    ) -> None:
        self._loader = loader

    def get(
        self,
        *,
        force_refresh: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> RuntimeSettings:
        if env is not None:
            return self._loader(env)
        return get_runtime_settings(force_refresh=force_refresh)

    def override(self, snapshot: RuntimeSettings | None) -> None:
        set_runtime_settings_override(snapshot)

    def clear(self) -> None:
        clear_runtime_settings_cache()


DEFAULT_RUNTIME_SETTINGS_PROVIDER = RuntimeSettingsProvider()


__all__ = [
    "DEFAULT_RUNTIME_SETTINGS_PROVIDER",
    "RuntimeSettings",
    "RuntimeSettingsProvider",
    "load_runtime_settings",
    "get_runtime_settings",
    "set_runtime_settings_override",
    "clear_runtime_settings_cache",
]

"""Immutable runtime settings snapshot helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any

from gpt_trader.app.config.bot_config import BotConfig


class FrozenConfigProxy(Mapping[str, Any]):
    """Read-only view of configuration data with attribute access."""

    __slots__ = ("_data",)

    def __init__(self, source: dict[str, Any]) -> None:
        object.__setattr__(
            self,
            "_data",
            MappingProxyType({k: _freeze_value(v) for k, v in sorted(source.items())}),
        )

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"FrozenConfigProxy has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name != "_data":
            raise AttributeError("FrozenConfigProxy is immutable")
        super().__setattr__(name, value)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()


def _freeze_dict(source: dict[str, Any]) -> FrozenConfigProxy:
    return FrozenConfigProxy(source)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, FrozenConfigProxy):
        return value
    if isinstance(value, MappingProxyType):
        return FrozenConfigProxy(dict(value))
    if isinstance(value, dict):
        return _freeze_dict(value)
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze_value(v) for v in value))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _freeze_value(asdict(value))
    return value


def _capture_environment() -> MappingProxyType[str, str]:
    env_items = sorted(os.environ.items())
    return MappingProxyType({key: value for key, value in env_items})


def _thaw_value(value: Any) -> Any:
    if isinstance(value, FrozenConfigProxy):
        return {k: _thaw_value(v) for k, v in value.items()}
    if isinstance(value, MappingProxyType):
        return {k: _thaw_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(element) for element in value]
    return value


def _prepare_for_serialization(value: Any) -> Any:
    if isinstance(value, FrozenConfigProxy):
        return {k: _prepare_for_serialization(v) for k, v in value.items()}
    if isinstance(value, MappingProxyType):
        return {k: _prepare_for_serialization(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_prepare_for_serialization(element) for element in value]
    return value


class RuntimeSettingsSnapshot:
    """Immutable snapshot of runtime-impacting configuration."""

    __slots__ = ("_config", "_env")

    def __init__(self, config_data: FrozenConfigProxy, env_vars: MappingProxyType[str, str]) -> None:
        self._config = config_data
        self._env = env_vars

    @property
    def config_data(self) -> FrozenConfigProxy:
        """Access the frozen config dictionary."""
        return self._config

    @property
    def env_vars(self) -> MappingProxyType[str, str]:
        """Access the captured environment variables."""
        return self._env

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable representation suitable for serialization."""
        return {"config": _thaw_value(self._config), "env_vars": dict(self._env)}

    def serialize(self) -> str:
        """Return a deterministic JSON string of the snapshot."""
        return json.dumps(
            {"config": _prepare_for_serialization(self._config), "env_vars": dict(self._env)},
            sort_keys=True,
            default=str,
        )

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"RuntimeSettingsSnapshot has no attribute '{name}'")

    def __getitem__(self, name: str) -> Any:
        return self._config[name]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._config

    def __iter__(self):
        return iter(self._config)


def create_runtime_settings_snapshot(config: BotConfig) -> RuntimeSettingsSnapshot:
    """Create a snapshot from a mutable BotConfig."""

    config_dict = asdict(config)
    frozen_config = _freeze_value(config_dict)
    env_snapshot = _capture_environment()
    return RuntimeSettingsSnapshot(config_data=frozen_config, env_vars=env_snapshot)


def ensure_runtime_settings_snapshot(
    config_or_snapshot: BotConfig | RuntimeSettingsSnapshot,
) -> RuntimeSettingsSnapshot:
    """Return a snapshot, creating one if only a BotConfig was provided."""

    if isinstance(config_or_snapshot, RuntimeSettingsSnapshot):
        return config_or_snapshot
    return create_runtime_settings_snapshot(config_or_snapshot)


__all__ = [
    "FrozenConfigProxy",
    "RuntimeSettingsSnapshot",
    "create_runtime_settings_snapshot",
    "ensure_runtime_settings_snapshot",
]

"""Shared environment and filesystem fixtures for integration-heavy unit tests."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet

from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings


@dataclass(slots=True)
class VaultSecretRecord:
    """Internal representation of stored secrets for the vault stub."""

    data: dict[str, Any]
    metadata: dict[str, Any]


class VaultKVv2Stub:
    """Subset of hvac's KV v2 API used by the SecretsManager."""

    def __init__(self, storage: dict[str, VaultSecretRecord]) -> None:
        self._storage = storage
        # Create real methods that can have side_effect set
        from unittest.mock import MagicMock

        self.create_or_update_secret = MagicMock(side_effect=self._create_or_update_secret)
        self.read_secret_version = MagicMock(side_effect=self._read_secret_version)
        self.delete_metadata_and_all_versions = MagicMock(
            side_effect=self._delete_metadata_and_all_versions
        )
        self.list_secrets = MagicMock(side_effect=self._list_secrets)

    def _create_or_update_secret(self, *, path: str, secret: dict[str, Any]) -> None:
        self._storage[path] = VaultSecretRecord(
            data=dict(secret),
            metadata={"created": True},
        )

    def _read_secret_version(self, *, path: str) -> dict[str, Any]:
        record = self._storage.get(path)
        if record is None:
            raise KeyError(path)
        return {"data": {"data": dict(record.data)}}

    def _delete_metadata_and_all_versions(self, *, path: str) -> None:
        self._storage.pop(path, None)

    def _list_secrets(self, *, path: str) -> dict[str, Any]:
        keys = [key for key in self._storage if key.startswith(path)]
        return {"data": {"keys": keys}}


class VaultSecretsStub:
    def __init__(self, storage: dict[str, VaultSecretRecord]) -> None:
        self.kv = type("KVNamespace", (), {"v2": VaultKVv2Stub(storage)})()


class VaultClientStub:
    """Minimal stand-in for hvac.Client with KV v2 support."""

    def __init__(self) -> None:
        self._storage: dict[str, VaultSecretRecord] = {}
        self.secrets = VaultSecretsStub(self._storage)
        self._authenticated = True

    def is_authenticated(self) -> bool:
        return self._authenticated

    def set_authenticated(self, value: bool) -> None:
        self._authenticated = value


@pytest.fixture
def hvac_stub(monkeypatch: pytest.MonkeyPatch) -> VaultClientStub:
    """Provide a stub hvac client and inject it into sys.modules."""

    client = VaultClientStub()

    class _ModuleShim:
        Client = lambda *_, **__: client  # noqa: E731 - simple factory shim

    monkeypatch.setitem(sys.modules, "hvac", _ModuleShim())
    return client


@pytest.fixture
def temp_home(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Point Path.home() to an isolated temporary directory."""
    home_dir = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(Path, "home", lambda: home_dir)
    return home_dir


@pytest.fixture
def env_override(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, str]], None]:
    """Apply environment overrides and restore afterwards."""

    def _apply(overrides: dict[str, str]) -> None:
        for key, value in overrides.items():
            monkeypatch.setenv(key, value)

    return _apply


@pytest.fixture
def runtime_settings_factory(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[..., RuntimeSettings]:
    """Build RuntimeSettings snapshots rooted in a temporary directory."""

    def _builder(
        *,
        event_store_override: bool = False,
        env_overrides: dict[str, str] | None = None,
    ) -> RuntimeSettings:
        runtime_root = tmp_path_factory.mktemp("runtime")
        env: dict[str, str] = {
            "GPT_TRADER_RUNTIME_ROOT": str(runtime_root),
            "BOT_V2_ENCRYPTION_KEY": Fernet.generate_key().decode(),
            "ENV": "development",
        }
        if event_store_override:
            event_root = tmp_path_factory.mktemp("event_store")
            env["EVENT_STORE_ROOT"] = str(event_root)
        if env_overrides:
            env.update(env_overrides)
        return load_runtime_settings(env)

    return _builder


@pytest.fixture
def patched_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
    runtime_settings_factory: Callable[..., RuntimeSettings],
) -> RuntimeSettings:
    """Override load_runtime_settings to return a deterministic snapshot."""
    settings = runtime_settings_factory()
    monkeypatch.setattr(
        "bot_v2.orchestration.storage.load_runtime_settings",
        lambda: settings,
    )
    monkeypatch.setattr(
        "bot_v2.orchestration.runtime_settings.load_runtime_settings",
        lambda env=None: settings if env is None else load_runtime_settings(env),
    )
    return settings


@pytest.fixture(scope="session")
def yahoo_provider_stub() -> Any:
    """Simple Yahoo provider stub returning injected data frames."""

    class _Provider:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.responses: dict[str, Any] = {}

        def queue_response(self, symbol: str, frame: Any) -> None:
            self.responses[symbol] = frame

        def get_historical_data(self, symbol: str, period: str) -> Any:
            self.calls.append({"symbol": symbol, "period": period})
            if symbol not in self.responses:
                raise ValueError(f"no data queued for {symbol}")
            return self.responses[symbol]

    return _Provider()


@contextmanager
def temporary_cwd(path: Path) -> Iterator[None]:
    """Temporarily change the working directory for file-based logic."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

"""Shared fixtures for secrets_manager tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet

from gpt_trader.config.runtime_settings import RuntimeSettings


@pytest.fixture
def secrets_runtime_settings(runtime_settings_factory) -> RuntimeSettings:
    """Runtime settings tailored for secrets_manager tests."""
    return runtime_settings_factory(
        env_overrides={
            "ENV": "development",
            "GPT_TRADER_ENCRYPTION_KEY": Fernet.generate_key().decode(),
        }
    )


@pytest.fixture
def deterministic_fernet() -> Fernet:
    """Deterministic Fernet instance for predictable tests."""
    # Use a fixed key for deterministic encryption/decryption
    test_key = b"Z9mB9nKp8sVqLyWsC5uE4oHj7gFdR2aL3xYiN6wTzQc="
    return Fernet(test_key)


@pytest.fixture
def patched_require_fernet(deterministic_fernet: Fernet, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch _require_fernet to return deterministic instance."""
    from gpt_trader.security import secrets_manager

    def mock_require_fernet() -> type[Fernet]:
        return type(deterministic_fernet)

    monkeypatch.setattr(secrets_manager, "_require_fernet", mock_require_fernet)


@pytest.fixture
def secrets_dir(tmp_path: Path) -> Path:
    """Temporary secrets directory for file-based storage tests."""
    secrets_dir = tmp_path / ".gpt_trader" / "secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    return secrets_dir


@pytest.fixture
def mock_secrets_storage(secrets_dir: Path) -> dict[str, dict[str, Any]]:
    """In-memory secrets storage for testing."""
    return {}


@pytest.fixture
def sample_secrets() -> dict[str, dict[str, Any]]:
    """Sample secrets data for testing."""
    return {
        "brokers/coinbase": {
            "api_key": "test-api-key-123",
            "secret": "test-secret-456",
            "passphrase": "test-passphrase",
        },
        "brokers/binance": {
            "api_key": "binance-api-key-789",
            "secret": "binance-secret-012",
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "testuser",
            "password": "testpassword",
        },
    }


@pytest.fixture
def hvac_stub(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Mock hvac client stub."""

    class MockHvacClient:
        def __init__(self):
            self._storage = {}
            self.secrets = MagicMock()
            self.secrets.kv.v2.create_or_update_secret.side_effect = self._create_or_update
            self.secrets.kv.v2.read_secret_version.side_effect = self._read
            self.secrets.kv.v2.delete_metadata_and_all_versions.side_effect = self._delete
            self.secrets.kv.v2.list_secrets.side_effect = self._list
            self._authenticated = True

        def is_authenticated(self):
            return self._authenticated

        def set_authenticated(self, value: bool):
            self._authenticated = value

        def _create_or_update(self, path, secret, **kwargs):
            # Wrap secret in an object with .data to match test expectations
            self._storage[path] = type("Record", (), {"data": secret})()
            return {"data": {"created_time": "now"}}

        def _read(self, path, **kwargs):
            if path in self._storage:
                stored_obj = self._storage[path]
                # Tests insert objects with .data, but _read returns nested dict structure
                secret_data = stored_obj.data if hasattr(stored_obj, "data") else stored_obj
                return {"data": {"data": secret_data}}
            raise Exception("Secret not found")

        def _delete(self, path, **kwargs):
            if path in self._storage:
                del self._storage[path]

        def _list(self, path, **kwargs):
            keys = list(self._storage.keys())
            return {"data": {"keys": keys}}

    mock_client = MockHvacClient()

    # Mock the hvac module
    mock_hvac = MagicMock()
    mock_hvac.Client.return_value = mock_client

    monkeypatch.setitem(sys.modules, "hvac", mock_hvac)

    return mock_client


@pytest.fixture
def secrets_manager_with_fallback(
    secrets_runtime_settings: RuntimeSettings,
    patched_require_fernet: None,
    monkeypatch: pytest.MonkeyPatch,
    secrets_dir: Path,
) -> Any:
    """SecretsManager instance with file fallback (no vault)."""
    from gpt_trader.security.secrets_manager import SecretsManager

    # Force vault to be disabled
    monkeypatch.setenv("VAULT_TOKEN", "")

    return SecretsManager(
        vault_enabled=False, settings=secrets_runtime_settings, secrets_dir=secrets_dir
    )


@pytest.fixture
def secrets_manager_with_vault(
    secrets_runtime_settings: RuntimeSettings,
    patched_require_fernet: None,
    hvac_stub: Any,
    monkeypatch: pytest.MonkeyPatch,
    secrets_dir: Path,
) -> Any:
    """SecretsManager instance with mocked vault."""
    from gpt_trader.config.runtime_settings import load_runtime_settings
    from gpt_trader.security.secrets_manager import SecretsManager

    # Enable vault with stub
    monkeypatch.setenv("VAULT_TOKEN", "test-token")
    monkeypatch.setenv("VAULT_ADDR", "http://vault.local")

    # Create new settings with the updated environment
    updated_settings = load_runtime_settings()

    return SecretsManager(vault_enabled=True, settings=updated_settings, secrets_dir=secrets_dir)


@pytest.fixture
def corrupted_file_data(secrets_dir: Path) -> Path:
    """Create a corrupted encrypted file for testing error handling."""
    corrupted_file = secrets_dir / "test_corrupted.enc"
    corrupted_file.write_bytes(b"corrupted-data-that-cannot-be-decrypted")
    return corrupted_file


@pytest.fixture
def mock_vault_failure(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock vault client that always fails."""
    vault_mock = MagicMock()
    vault_mock.is_authenticated.return_value = False
    vault_mock.secrets.kv.v2.create_or_update_secret.side_effect = Exception("Vault error")
    vault_mock.secrets.kv.v2.read_secret_version.side_effect = Exception("Vault error")
    vault_mock.secrets.kv.v2.delete_metadata_and_all_versions.side_effect = Exception("Vault error")
    vault_mock.secrets.kv.v2.list_secrets.side_effect = Exception("Vault error")

    return vault_mock


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Advance time deterministically without real sleep."""
    import time

    current = {"value": time.time()}

    def advance(seconds: float) -> None:
        current["value"] += seconds

    monkeypatch.setattr(time, "time", lambda: current["value"])
    monkeypatch.setattr(time, "sleep", lambda seconds: advance(seconds))

    return advance

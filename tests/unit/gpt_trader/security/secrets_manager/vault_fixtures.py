from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet

from gpt_trader.app.config import BotConfig


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
def secrets_manager_with_vault(
    secrets_bot_config: BotConfig,
    patched_require_fernet: None,
    hvac_stub: Any,
    monkeypatch: pytest.MonkeyPatch,
    secrets_dir: Path,
) -> Any:
    """SecretsManager instance with mocked vault."""
    from gpt_trader.security.secrets_manager import SecretsManager

    # Set required environment variables for SecretsManager
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("GPT_TRADER_ENCRYPTION_KEY", Fernet.generate_key().decode())

    # Enable vault with stub
    monkeypatch.setenv("VAULT_TOKEN", "test-token")
    monkeypatch.setenv("VAULT_ADDR", "http://vault.local")

    return SecretsManager(vault_enabled=True, config=secrets_bot_config, secrets_dir=secrets_dir)


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

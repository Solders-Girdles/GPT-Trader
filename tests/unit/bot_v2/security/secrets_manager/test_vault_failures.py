"""Vault failure scenarios for SecretsManager."""

from __future__ import annotations

from typing import Any

import pytest

from bot_v2.security.secrets_manager import SecretsManager


class TestVaultFailures:
    """Ensure Vault errors produce safe fallbacks and logging."""

    def test_vault_authentication_failure(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        hvac_stub: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hvac_stub.set_authenticated(False)
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        monkeypatch.setenv("VAULT_ADDR", "http://vault.local")

        manager = SecretsManager(vault_enabled=True, settings=secrets_runtime_settings)

        assert manager._vault_enabled is False
        assert manager._vault_client is None

    def test_vault_store_failure(
        self,
        secrets_manager_with_vault: Any,
        sample_secrets: dict[str, dict[str, Any]],
        hvac_stub: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        manager = secrets_manager_with_vault
        secret_data = sample_secrets["brokers/coinbase"]
        hvac_stub.secrets.kv.v2.create_or_update_secret.side_effect = Exception("Vault error")

        with caplog.at_level("ERROR"):
            result = manager.store_secret("test/secret", secret_data)

        assert result is False
        assert any("Failed to store secret" in message for message in caplog.messages)

    def test_vault_retrieve_failure(
        self, secrets_manager_with_vault: Any, hvac_stub: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        manager = secrets_manager_with_vault
        hvac_stub.secrets.kv.v2.read_secret_version.side_effect = Exception("Vault error")

        with caplog.at_level("ERROR"):
            assert manager.get_secret("test/secret") is None

        assert any("Failed to retrieve secret" in message for message in caplog.messages)

    def test_vault_list_failure(
        self, secrets_manager_with_vault: Any, hvac_stub: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        manager = secrets_manager_with_vault
        hvac_stub.secrets.kv.v2.list_secrets.side_effect = Exception("Vault error")

        with caplog.at_level("WARNING"):
            listed = manager.list_secrets()

        assert listed == []
        assert any("Unable to list secrets from vault" in message for message in caplog.messages)

    def test_vault_delete_failure(
        self, secrets_manager_with_vault: Any, hvac_stub: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        manager = secrets_manager_with_vault
        hvac_stub.secrets.kv.v2.delete_metadata_and_all_versions.side_effect = Exception(
            "Vault error"
        )

        with caplog.at_level("ERROR"):
            result = manager.delete_secret("test/secret")

        assert result is False
        assert any("Failed to delete secret" in message for message in caplog.messages)

    def test_vault_environment_configuration(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        hvac_stub: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("VAULT_TOKEN", "env-token")
        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")

        from bot_v2.orchestration.runtime_settings import load_runtime_settings

        updated_settings = load_runtime_settings()
        manager = SecretsManager(vault_enabled=True, settings=updated_settings)

        assert manager._vault_enabled is True
        assert manager._vault_client is not None

    def test_vault_missing_environment_variables(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("VAULT_TOKEN", raising=False)
        monkeypatch.delenv("VAULT_ADDR", raising=False)

        manager = SecretsManager(vault_enabled=True, settings=secrets_runtime_settings)

        assert manager._vault_enabled is False
        assert manager._vault_client is None

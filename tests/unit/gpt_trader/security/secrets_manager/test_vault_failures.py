"""Vault failure scenarios for SecretsManager."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpt_trader.security.secrets_manager import (
    FILE_SECRET_FALLBACK_ENVIRONMENT_VARIABLE,
    SecretsManager,
    VaultConfigurationError,
)


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

        manager = SecretsManager(vault_enabled=True, config=secrets_runtime_settings)

        # When auth fails, vault_enabled is set to False to fall back to file storage.
        # The client reference may still exist but won't be used.
        assert manager._vault_enabled is False

    def test_production_missing_vault_token_fails_closed(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        hvac_stub: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
        monkeypatch.delenv("VAULT_TOKEN", raising=False)

        with pytest.raises(VaultConfigurationError, match="VAULT_TOKEN"):
            SecretsManager(vault_enabled=True, config=secrets_runtime_settings)

    def test_production_vault_authentication_failure_fails_closed(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        hvac_stub: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hvac_stub.set_authenticated(False)
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")

        with pytest.raises(VaultConfigurationError, match="Vault authentication failed"):
            SecretsManager(vault_enabled=True, config=secrets_runtime_settings)

    def test_production_requires_https_vault_addr(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        hvac_stub: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        monkeypatch.setenv("VAULT_ADDR", "http://vault.local")

        with pytest.raises(VaultConfigurationError, match="https://"):
            SecretsManager(vault_enabled=True, config=secrets_runtime_settings)

    def test_production_file_fallback_requires_explicit_opt_in(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        hvac_stub: Any,
        monkeypatch: pytest.MonkeyPatch,
        secrets_dir: Path,
    ) -> None:
        hvac_stub.set_authenticated(False)
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")
        monkeypatch.setenv(FILE_SECRET_FALLBACK_ENVIRONMENT_VARIABLE, "1")

        manager = SecretsManager(
            vault_enabled=True,
            config=secrets_runtime_settings,
            secrets_dir=secrets_dir,
        )

        assert manager._vault_enabled is False
        assert manager.store_secret("test/secret", {"key": "value"}) is True
        assert (secrets_dir / "test_secret.enc").exists()

    @pytest.mark.parametrize(
        ("config_environment", "process_environment", "vault_address", "expected_match"),
        (
            ("production", None, None, "VAULT_ADDR must be set outside development"),
            ("canary", "development", "http://vault.local", "https://"),
        ),
    )
    def test_config_environment_drives_non_development_vault_fail_closed(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        hvac_stub: Any,
        monkeypatch: pytest.MonkeyPatch,
        config_environment: str,
        process_environment: str | None,
        vault_address: str | None,
        expected_match: str,
    ) -> None:
        config = secrets_runtime_settings.model_copy(update={"environment": config_environment})
        if process_environment is None:
            monkeypatch.delenv("ENV", raising=False)
        else:
            monkeypatch.setenv("ENV", process_environment)
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        if vault_address is None:
            monkeypatch.delenv("VAULT_ADDR", raising=False)
        else:
            monkeypatch.setenv("VAULT_ADDR", vault_address)

        with pytest.raises(VaultConfigurationError, match=expected_match) as error:
            SecretsManager(vault_enabled=True, config=config)

        assert f"environment '{config_environment}'" in str(error.value)

    @pytest.mark.parametrize("config_environment", ("test", "local"))
    def test_config_development_alias_allows_ephemeral_key_and_vault_fallback(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        hvac_stub: Any,
        monkeypatch: pytest.MonkeyPatch,
        config_environment: str,
    ) -> None:
        config = secrets_runtime_settings.model_copy(update={"environment": config_environment})
        monkeypatch.setenv("ENV", "production")
        monkeypatch.delenv("GPT_TRADER_ENCRYPTION_KEY", raising=False)
        monkeypatch.delenv("VAULT_ADDR", raising=False)
        monkeypatch.delenv("VAULT_TOKEN", raising=False)

        manager = SecretsManager(vault_enabled=True, config=config)

        assert manager._environment == config_environment
        assert manager._encryption_key_is_ephemeral is True
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
        # Set development environment for test
        monkeypatch.setenv("ENV", "development")
        monkeypatch.setenv("VAULT_TOKEN", "env-token")
        monkeypatch.setenv("VAULT_ADDR", "https://vault.example.com")

        manager = SecretsManager(vault_enabled=True, config=secrets_runtime_settings)

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

        manager = SecretsManager(vault_enabled=True, config=secrets_runtime_settings)

        assert manager._vault_enabled is False
        assert manager._vault_client is None

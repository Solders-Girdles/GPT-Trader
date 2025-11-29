"""Vault-related error handling for SecretsManager."""

from __future__ import annotations

from typing import Any


class TestVaultErrors:
    """Ensure vault failures degrade gracefully to file storage."""

    def test_vault_connection_error(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        monkeypatch: Any,
    ) -> None:
        import sys

        class FailingVaultClient:
            def __init__(self, *args, **kwargs):
                raise ConnectionError("Vault connection failed")

        vault_module = type("VaultModule", (), {"Client": FailingVaultClient})
        monkeypatch.setitem(sys.modules, "hvac", vault_module)

        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        monkeypatch.setenv("VAULT_ADDR", "http://vault.local")

        from gpt_trader.security.secrets_manager import SecretsManager

        manager = SecretsManager(vault_enabled=True, config=secrets_runtime_settings)
        assert manager._vault_enabled is False

    def test_vault_authentication_error(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        hvac_stub: Any,
        monkeypatch: Any,
    ) -> None:
        hvac_stub.set_authenticated(False)
        monkeypatch.setenv("VAULT_TOKEN", "invalid-token")
        monkeypatch.setenv("VAULT_ADDR", "http://vault.local")

        from gpt_trader.security.secrets_manager import SecretsManager

        manager = SecretsManager(vault_enabled=True, config=secrets_runtime_settings)
        assert manager._vault_enabled is False

    def test_vault_operation_timeout(
        self, secrets_manager_with_vault: Any, hvac_stub: Any, monkeypatch: Any
    ) -> None:
        manager = secrets_manager_with_vault
        hvac_stub.secrets.kv.v2.create_or_update_secret.side_effect = TimeoutError("Vault timeout")

        assert manager.store_secret("test/timeout", {"key": "value"}) is False

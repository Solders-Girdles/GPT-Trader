"""Basic Vault integration tests for SecretsManager."""

from __future__ import annotations

from typing import Any


class TestVaultBasics:
    """Verify happy-path Vault operations."""

    def test_vault_store_success(
        self,
        secrets_manager_with_vault: Any,
        sample_secrets: dict[str, dict[str, Any]],
        hvac_stub: Any,
    ) -> None:
        manager = secrets_manager_with_vault
        secret_path = "brokers/coinbase"
        secret_data = sample_secrets[secret_path]

        assert manager.store_secret(secret_path, secret_data) is True
        assert secret_path in hvac_stub._storage
        assert hvac_stub._storage[secret_path].data == secret_data

    def test_vault_retrieve_success(
        self,
        secrets_manager_with_vault: Any,
        sample_secrets: dict[str, dict[str, Any]],
        hvac_stub: Any,
    ) -> None:
        manager = secrets_manager_with_vault
        secret_path = "brokers/binance"
        secret_data = sample_secrets[secret_path]
        hvac_stub._storage[secret_path] = type("Record", (), {"data": secret_data})()

        assert manager.get_secret(secret_path) == secret_data

    def test_vault_delete_success(self, secrets_manager_with_vault: Any, hvac_stub: Any) -> None:
        manager = secrets_manager_with_vault
        secret_path = "test/secret"
        hvac_stub._storage[secret_path] = type("Record", (), {"data": {"key": "value"}})()

        assert manager.delete_secret(secret_path) is True
        assert secret_path not in hvac_stub._storage

    def test_vault_list_success(self, secrets_manager_with_vault: Any, hvac_stub: Any) -> None:
        manager = secrets_manager_with_vault
        hvac_stub._storage.update(
            {
                "secret1": type("Record", (), {"data": {}})(),
                "secret2": type("Record", (), {"data": {}})(),
                "secret3": type("Record", (), {"data": {}})(),
            }
        )

        listed = manager.list_secrets()
        assert len(listed) >= 3
        for key in ("secret1", "secret2", "secret3"):
            assert key in listed

    def test_vault_retrieve_missing_secret(self, secrets_manager_with_vault: Any) -> None:
        assert secrets_manager_with_vault.get_secret("nonexistent/secret") is None

    def test_vault_delete_missing_secret(self, secrets_manager_with_vault: Any) -> None:
        assert secrets_manager_with_vault.delete_secret("nonexistent/secret") is True

    def test_vault_caching_behavior(
        self,
        secrets_manager_with_vault: Any,
        sample_secrets: dict[str, dict[str, Any]],
        hvac_stub: Any,
    ) -> None:
        manager = secrets_manager_with_vault
        secret_path = "test/cache"
        secret_data = sample_secrets["brokers/coinbase"]

        manager.store_secret(secret_path, secret_data)

        assert manager.get_secret(secret_path) == secret_data
        assert manager.get_secret(secret_path) == secret_data  # cache hit

        manager.clear_cache()
        assert manager.get_secret(secret_path) == secret_data

    def test_vault_disabled_fallback(
        self, secrets_manager_with_fallback: Any, sample_secrets: dict[str, dict[str, Any]]
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/fallback"
        secret_data = sample_secrets["brokers/coinbase"]

        assert manager.store_secret(secret_path, secret_data) is True
        assert manager.get_secret(secret_path) == secret_data

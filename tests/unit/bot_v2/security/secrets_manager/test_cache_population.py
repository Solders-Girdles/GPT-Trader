"""Cache population and invalidation tests for SecretsManager."""

from __future__ import annotations

from typing import Any


class TestCachePopulation:
    """Verify cache warm-up and invalidation behaviours."""

    def test_cache_population_on_retrieve(
        self, secrets_manager_with_fallback: Any, sample_secrets: dict[str, dict[str, Any]]
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "brokers/coinbase"
        secret_data = sample_secrets[secret_path]

        manager._store_to_file(secret_path, secret_data)
        assert len(manager._secrets_cache) == 0

        retrieved = manager.get_secret(secret_path)

        assert len(manager._secrets_cache) == 1
        assert secret_path in manager._secrets_cache
        assert manager._secrets_cache[secret_path] == secret_data
        assert retrieved == secret_data

    def test_cache_hit_on_subsequent_retrieve(
        self, secrets_manager_with_fallback: Any, sample_secrets: dict[str, dict[str, Any]]
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "brokers/binance"
        secret_data = sample_secrets[secret_path]

        manager.store_secret(secret_path, secret_data)

        retrieved1 = manager.get_secret(secret_path)
        assert retrieved1 == secret_data
        assert len(manager._secrets_cache) == 1

        retrieved2 = manager.get_secret(secret_path)
        assert retrieved2 == secret_data
        assert len(manager._secrets_cache) == 1

    def test_cache_invalidate_on_store(
        self, secrets_manager_with_fallback: Any, sample_secrets: dict[str, dict[str, Any]]
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/invalidate"

        initial_data = {"key": "initial"}
        manager.store_secret(secret_path, initial_data)
        assert manager.get_secret(secret_path) == initial_data

        new_data = {"key": "new"}
        manager.store_secret(secret_path, new_data)

        assert manager._secrets_cache[secret_path] == new_data
        assert manager.get_secret(secret_path) == new_data

    def test_cache_clear_all(
        self, secrets_manager_with_fallback: Any, sample_secrets: dict[str, dict[str, Any]]
    ) -> None:
        manager = secrets_manager_with_fallback

        for path, data in sample_secrets.items():
            manager.store_secret(path, data)
            manager.get_secret(path)

        assert len(manager._secrets_cache) >= len(sample_secrets)

        manager.clear_cache()
        assert len(manager._secrets_cache) == 0

    def test_cache_invalidate_on_delete(
        self, secrets_manager_with_fallback: Any, sample_secrets: dict[str, dict[str, Any]]
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/delete"
        secret_data = sample_secrets["brokers/coinbase"]

        manager.store_secret(secret_path, secret_data)
        manager.get_secret(secret_path)
        assert secret_path in manager._secrets_cache

        result = manager.delete_secret(secret_path)

        assert result is True
        assert secret_path not in manager._secrets_cache

    def test_cache_missing_secret(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback

        retrieved = manager.get_secret("nonexistent/secret")

        assert retrieved is None
        assert len(manager._secrets_cache) == 0

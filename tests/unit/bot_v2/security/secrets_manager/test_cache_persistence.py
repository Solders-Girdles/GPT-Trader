"""Cache persistence and lifecycle tests for SecretsManager."""

from __future__ import annotations

from typing import Any


class TestCachePersistence:
    """Validate cache lifecycle across reset and update operations."""

    def test_cache_with_clear_and_repopulate(
        self, secrets_manager_with_fallback: Any, sample_secrets: dict[str, dict[str, Any]]
    ) -> None:
        manager = secrets_manager_with_fallback

        for path, data in list(sample_secrets.items())[:2]:
            manager.store_secret(path, data)
            manager.get_secret(path)

        initial_cache_size = len(manager._secrets_cache)
        assert initial_cache_size >= 2

        manager.clear_cache()
        assert len(manager._secrets_cache) == 0

        for path in list(sample_secrets.keys())[:2]:
            assert manager.get_secret(path) is not None

        assert len(manager._secrets_cache) == initial_cache_size

    def test_cache_persistence_across_operations(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/persistence"
        secret_data = {"key": "initial_value"}

        manager.store_secret(secret_path, secret_data)
        assert manager.get_secret(secret_path) == secret_data

        updated_data = {"key": "updated_value"}
        manager.store_secret(secret_path, updated_data)

        assert manager._secrets_cache[secret_path] == updated_data
        assert manager.get_secret(secret_path) == updated_data

        if "_key" in updated_data:
            manager.rotate_key(secret_path)
            assert secret_path in manager._secrets_cache
            rotated_data = manager._secrets_cache[secret_path]
            assert "_rotated_at" in rotated_data

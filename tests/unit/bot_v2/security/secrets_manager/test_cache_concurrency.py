"""Concurrency and isolation tests for SecretsManager cache."""

from __future__ import annotations

from typing import Any

from bot_v2.security.secrets_manager import SecretsManager


class TestCacheConcurrency:
    """Ensure cache behaves correctly under concurrent access."""

    def test_cache_concurrent_access(
        self, secrets_manager_with_fallback: Any, sample_secrets: dict[str, dict[str, Any]]
    ) -> None:
        import threading

        manager = secrets_manager_with_fallback
        secret_path = "test/concurrent"
        secret_data = sample_secrets["brokers/coinbase"]

        manager.store_secret(secret_path, secret_data)

        results: list[dict[str, Any] | None] = []

        def retrieve_secret() -> None:
            results.append(manager.get_secret(secret_path))

        threads = [threading.Thread(target=retrieve_secret) for _ in range(5)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 5
        assert all(result == secret_data for result in results)
        assert len(manager._secrets_cache) == 1
        assert manager._secrets_cache[secret_path] == secret_data

    def test_cache_isolation_between_managers(
        self,
        secrets_manager_with_fallback: Any,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
    ) -> None:
        manager1 = secrets_manager_with_fallback
        manager2 = SecretsManager(
            settings=secrets_runtime_settings,
            vault_enabled=False,
            secrets_dir=manager1._secrets_dir,
        )

        secret_path = "test/isolation"
        secret_data = {"key": "value"}

        manager1.store_secret(secret_path, secret_data)
        manager1.get_secret(secret_path)

        assert secret_path in manager1._secrets_cache
        assert secret_path not in manager2._secrets_cache

        retrieved2 = manager2.get_secret(secret_path)
        assert retrieved2 == secret_data

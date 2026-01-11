"""Concurrency and isolation tests for SecretsManager cache."""

from __future__ import annotations

from typing import Any

from gpt_trader.security.secrets_manager import SecretsManager


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
            thread.join(timeout=2)
            assert not thread.is_alive()

        assert len(results) == 5
        assert all(result == secret_data for result in results)
        assert len(manager._secrets_cache) == 1
        assert manager._secrets_cache[secret_path] == secret_data

    def test_cache_isolation_between_managers(
        self,
        secrets_manager_with_fallback: Any,
        secrets_bot_config: Any,
        patched_require_fernet: None,
    ) -> None:
        # Note: Don't use secrets_runtime_settings here as it sets a NEW encryption key,
        # but manager1 was already created with a different key from secrets_manager_with_fallback.
        # Both managers must use the same encryption key to read each other's files.
        manager1 = secrets_manager_with_fallback
        manager2 = SecretsManager(
            vault_enabled=False,
            config=secrets_bot_config,
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

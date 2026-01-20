"""Key rotation scenario tests for SecretsManager."""

from __future__ import annotations

from typing import Any

import pytest


class TestKeyRotationScenarios:
    """Cover vault, concurrency, and complex secret scenarios."""

    def test_rotate_key_with_vault_storage(
        self,
        secrets_manager_with_vault: Any,
        sample_secrets: dict[str, dict[str, Any]],
        hvac_stub: Any,
    ) -> None:
        manager = secrets_manager_with_vault
        secret_path = "brokers/coinbase"
        secret_data = sample_secrets[secret_path]
        secret_data["_key"] = "old_key"

        manager.store_secret(secret_path, secret_data)

        assert manager.rotate_key(secret_path) is True
        vault_data = hvac_stub._storage[secret_path].data
        assert vault_data["_key"] != "old_key"

    def test_rotate_key_concurrent_access(self, secrets_manager_with_fallback: Any) -> None:
        import threading

        manager = secrets_manager_with_fallback
        secret_path = "test/concurrent"
        manager.store_secret(secret_path, {"_key": "old_key"})

        results: list[bool] = []

        def rotate_key() -> None:
            results.append(manager.rotate_key(secret_path))

        threads = [threading.Thread(target=rotate_key) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()

        assert all(results)
        assert "_key" in manager.get_secret(secret_path)

    def test_rotate_key_error_handling(
        self, secrets_manager_with_fallback: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/error"
        manager.store_secret(secret_path, {"_key": "old_key"})

        original_store = manager.store_secret

        def failing_store(path, data):
            if path == secret_path and "_key" in data and data["_key"] != "old_key":
                raise Exception("Storage error")
            return original_store(path, data)

        monkeypatch.setattr(manager, "store_secret", failing_store)

        assert manager.rotate_key(secret_path) is False

    def test_rotate_key_with_cache(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/cache"
        manager.store_secret(secret_path, {"_key": "old_key"})
        assert manager.get_secret(secret_path)["_key"] == "old_key"

        assert manager.rotate_key(secret_path) is True
        assert manager.get_secret(secret_path)["_key"] != "old_key"

    def test_rotate_key_idempotent(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/idempotent"
        manager.store_secret(secret_path, {"_key": "old_key"})

        assert manager.rotate_key(secret_path) is True
        first_key = manager.get_secret(secret_path)["_key"]

        assert manager.rotate_key(secret_path) is True
        second_key = manager.get_secret(secret_path)["_key"]
        assert second_key != first_key

    def test_rotate_key_with_complex_secret(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/complex"
        secret_data = {
            "_key": "old_key",
            "credentials": {
                "username": "testuser",
                "password": "testpass",
                "nested": {"deep_key": "deep_value"},
            },
            "settings": [
                {"name": "setting1", "value": "value1"},
                {"name": "setting2", "value": "value2"},
            ],
        }

        manager.store_secret(secret_path, secret_data)

        assert manager.rotate_key(secret_path) is True

        retrieved = manager.get_secret(secret_path)
        assert retrieved["credentials"]["username"] == "testuser"
        assert retrieved["credentials"]["nested"]["deep_key"] == "deep_value"
        assert len(retrieved["settings"]) == 2
        assert retrieved["_key"] != "old_key"

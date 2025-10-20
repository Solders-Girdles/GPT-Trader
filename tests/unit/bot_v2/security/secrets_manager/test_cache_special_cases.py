"""Special-case cache behaviours for SecretsManager."""

from __future__ import annotations

from typing import Any


class TestCacheSpecialCases:
    """Exercise vault-backed caching and diverse payload handling."""

    def test_cache_with_key_rotation(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/rotate"
        secret_data = {"_key": "old_key"}

        manager.store_secret(secret_path, secret_data)
        assert manager.get_secret(secret_path)["_key"] == "old_key"

        result = manager.rotate_key(secret_path)

        assert result is True
        assert secret_path in manager._secrets_cache
        assert manager._secrets_cache[secret_path]["_key"] != "old_key"

    def test_cache_with_vault_storage(
        self,
        secrets_manager_with_vault: Any,
        sample_secrets: dict[str, dict[str, Any]],
        hvac_stub: Any,
    ) -> None:
        manager = secrets_manager_with_vault
        secret_path = "test/vault_cache"
        secret_data = sample_secrets["brokers/coinbase"]

        manager.store_secret(secret_path, secret_data)

        assert secret_path in manager._secrets_cache
        assert manager._secrets_cache[secret_path] == secret_data
        assert manager.get_secret(secret_path) == secret_data

    def test_cache_with_different_data_types(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback

        test_cases = [
            ("string", {"value": "test_string"}),
            ("integer", {"value": 42}),
            ("float", {"value": 3.14}),
            ("boolean", {"value": True}),
            ("null", {"value": None}),
            ("list", {"value": [1, 2, 3]}),
            ("nested", {"value": {"inner": {"deep": "value"}}}),
        ]

        for name, data in test_cases:
            secret_path = f"test/types/{name}"
            manager.store_secret(secret_path, data)
            retrieved = manager.get_secret(secret_path)

            assert retrieved == data
            assert manager._secrets_cache[secret_path] == data

    def test_cache_memory_efficiency(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        large_data = {"data": "x" * 10_000}
        secret_path = "test/large"

        manager.store_secret(secret_path, large_data)
        retrieved = manager.get_secret(secret_path)

        assert retrieved == large_data
        assert manager._secrets_cache[secret_path] == large_data
        assert len(manager._secrets_cache) == 1

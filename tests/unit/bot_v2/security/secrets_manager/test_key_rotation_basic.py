"""Basic key rotation tests for SecretsManager."""

from __future__ import annotations

from typing import Any


class TestKeyRotationBasics:
    """Validate primary key rotation behaviours."""

    def test_rotate_key_success(
        self, secrets_manager_with_fallback: Any, sample_secrets: dict[str, dict[str, Any]]
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "brokers/coinbase"
        secret_data = sample_secrets[secret_path]

        manager.store_secret(secret_path, secret_data)
        assert manager.rotate_key(secret_path) is True

        retrieved = manager.get_secret(secret_path)
        assert retrieved is not None
        assert "_key" in retrieved
        assert retrieved["_key"] != "old"

    def test_rotate_key_missing_secret(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        assert manager.rotate_key("nonexistent/secret") is False

    def test_rotate_key_without_key_field(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/no_key"
        manager.store_secret(secret_path, {"other_field": "value"})

        assert manager.rotate_key(secret_path) is True
        retrieved = manager.get_secret(secret_path)
        assert retrieved is not None
        assert retrieved["other_field"] == "value"
        assert "_key" in retrieved
        assert "_rotated_at" in retrieved

    def test_rotate_key_adds_timestamp(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        manager.store_secret("test/timestamp", {"_key": "old_key"})

        assert manager.rotate_key("test/timestamp") is True
        assert "_rotated_at" in manager.get_secret("test/timestamp")

    def test_rotate_key_generates_new_key(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        manager.store_secret("test/new_key", {"_key": "old_key"})

        assert manager.rotate_key("test/new_key") is True
        assert manager.get_secret("test/new_key")["_key"] != "old_key"

    def test_rotate_key_preserves_other_fields(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/preserve"
        secret_data = {
            "_key": "old_key",
            "api_key": "test_api_key",
            "secret": "test_secret",
            "config": {"setting1": "value1", "setting2": "value2"},
        }

        manager.store_secret(secret_path, secret_data)

        assert manager.rotate_key(secret_path) is True

        retrieved = manager.get_secret(secret_path)
        assert retrieved["api_key"] == "test_api_key"
        assert retrieved["secret"] == "test_secret"
        assert retrieved["config"] == {"setting1": "value1", "setting2": "value2"}

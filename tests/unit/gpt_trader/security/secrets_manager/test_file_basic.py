"""Basic file storage behaviours for SecretsManager."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.config import path_registry


class TestFileBasics:
    """Verify core CRUD operations on encrypted files."""

    def test_file_store_success(
        self,
        secrets_manager_with_fallback: Any,
        sample_secrets: dict[str, dict[str, Any]],
        secrets_dir: Path,
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "brokers/coinbase"
        secret_data = sample_secrets[secret_path]

        assert manager.store_secret(secret_path, secret_data) is True

        expected_file = secrets_dir / "brokers_coinbase.enc"
        assert expected_file.exists()
        encrypted_data = expected_file.read_bytes()
        assert len(encrypted_data) > 0
        assert encrypted_data != json.dumps(secret_data).encode()

    def test_ephemeral_development_key_rejects_file_storage_across_instances(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patched_require_fernet: None,
        secrets_bot_config: BotConfig,
        secrets_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from gpt_trader.security.secrets_manager import SecretsManager

        monkeypatch.setenv("ENV", "development")
        monkeypatch.delenv("GPT_TRADER_ENCRYPTION_KEY", raising=False)

        secret_path = "brokers/coinbase"
        secret_data = {"api_key": "dummy-api-key"}

        first_manager = SecretsManager(
            vault_enabled=False,
            config=secrets_bot_config,
            secrets_dir=secrets_dir,
        )
        with caplog.at_level("ERROR"):
            assert first_manager.store_secret(secret_path, secret_data) is False

        expected_file = secrets_dir / "brokers_coinbase.enc"
        assert not expected_file.exists()
        assert "Failed to store secret" in caplog.messages
        assert any(
            getattr(record, "error_type", None) == "RuntimeError" for record in caplog.records
        )
        assert "dummy-api-key" not in caplog.text

        second_manager = SecretsManager(
            vault_enabled=False,
            config=secrets_bot_config,
            secrets_dir=secrets_dir,
        )
        assert second_manager.get_secret(secret_path) is None

    def test_file_retrieve_success(
        self, secrets_manager_with_fallback: Any, sample_secrets: dict[str, dict[str, Any]]
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "brokers/binance"
        secret_data = sample_secrets[secret_path]

        manager.store_secret(secret_path, secret_data)

        assert manager.get_secret(secret_path) == secret_data

    def test_file_delete_success(
        self,
        secrets_manager_with_fallback: Any,
        sample_secrets: dict[str, dict[str, Any]],
        secrets_dir: Path,
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/secret"
        secret_data = sample_secrets["brokers/coinbase"]

        manager.store_secret(secret_path, secret_data)
        expected_file = secrets_dir / "test_secret.enc"
        assert expected_file.exists()

        assert manager.delete_secret(secret_path) is True
        assert not expected_file.exists()

    def test_file_list_success(
        self,
        secrets_manager_with_fallback: Any,
        sample_secrets: dict[str, dict[str, Any]],
        secrets_dir: Path,
    ) -> None:
        manager = secrets_manager_with_fallback

        for path, data in sample_secrets.items():
            manager.store_secret(path, data)

        listed = manager.list_secrets()
        assert len(listed) >= len(sample_secrets)
        for path in sample_secrets:
            assert path in listed

    def test_file_retrieve_missing_secret(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        assert manager.get_secret("nonexistent/secret") is None

    def test_file_delete_missing_secret(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        assert manager.delete_secret("nonexistent/secret") is True

    def test_file_path_sanitization(
        self, secrets_manager_with_fallback: Any, secrets_dir: Path
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/complex/path/with/many/levels"

        manager.store_secret(secret_path, {"key": "value"})

        expected_file = secrets_dir / "test_complex_path_with_many_levels.enc"
        assert expected_file.exists()

    def test_file_directory_creation(
        self, secrets_manager_with_fallback: Any, tmp_path: Path
    ) -> None:
        os.environ["HOME"] = str(tmp_path)

        manager = secrets_manager_with_fallback
        manager.store_secret("test/directory", {"key": "value"})

        secrets_dir = tmp_path / ".gpt_trader" / "secrets"
        assert secrets_dir.exists() and secrets_dir.is_dir()

    def test_default_file_storage_uses_registry_secret_dir(
        self,
        secrets_bot_config: BotConfig,
        patched_require_fernet: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from cryptography.fernet import Fernet

        from gpt_trader.security.secrets_manager import SecretsManager

        default_dir = tmp_path / ".gpt_trader" / "secrets"
        monkeypatch.setattr(path_registry, "USER_SECRETS_DIR", default_dir)
        monkeypatch.setenv("VAULT_TOKEN", "")
        monkeypatch.setenv("ENV", "development")
        monkeypatch.setenv("GPT_TRADER_ENCRYPTION_KEY", Fernet.generate_key().decode())

        manager = SecretsManager(vault_enabled=False, config=secrets_bot_config)
        manager.store_secret("test/default", {"key": "value"})

        assert manager._secrets_dir == default_dir
        assert (default_dir / "test_default.enc").exists()

    def test_file_encryption_roundtrip(
        self, secrets_manager_with_fallback: Any, sample_secrets: dict[str, dict[str, Any]]
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/encryption"
        secret_data = sample_secrets["database"]

        manager.store_secret(secret_path, secret_data)
        retrieved = manager.get_secret(secret_path)

        assert retrieved == secret_data
        assert retrieved["host"] == "localhost"
        assert retrieved["port"] == 5432

    def test_file_overwrite_existing(
        self, secrets_manager_with_fallback: Any, secrets_dir: Path
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/overwrite"

        manager.store_secret(secret_path, {"key": "initial_value"})
        manager.store_secret(secret_path, {"key": "new_value", "additional": "data"})

        assert manager.get_secret(secret_path) == {"key": "new_value", "additional": "data"}

    def test_file_invalid_path_characters(
        self, secrets_manager_with_fallback: Any, secrets_dir: Path
    ) -> None:
        manager = secrets_manager_with_fallback

        test_paths = [
            "test/simple",
            "test-with-dash",
            "test_with_underscore",
            "test.with.dots",
            "test/with/many/levels",
        ]

        for path in test_paths:
            secret_data = {"path": path}
            manager.store_secret(path, secret_data)
            assert manager.get_secret(path) == secret_data

    def test_file_permissions_handling(
        self, secrets_manager_with_fallback: Any, secrets_dir: Path
    ) -> None:
        manager = secrets_manager_with_fallback
        secret_path = "test/permissions"

        manager.store_secret(secret_path, {"key": "value"})

        expected_file = secrets_dir / "test_permissions.enc"
        assert expected_file.exists()
        assert len(expected_file.read_bytes()) > 0

    def test_file_storage_with_different_data_types(
        self, secrets_manager_with_fallback: Any
    ) -> None:
        manager = secrets_manager_with_fallback
        test_data = {
            "string": "test_string",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"inner_key": "inner_value"},
        }

        manager.store_secret("test/types", test_data)
        assert manager.get_secret("test/types") == test_data

    def test_file_storage_empty_secret(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback

        manager.store_secret("test/empty", {})
        assert manager.get_secret("test/empty") == {}

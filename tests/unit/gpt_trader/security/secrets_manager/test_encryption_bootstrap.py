"""Tests for encryption bootstrap in SecretsManager."""

from __future__ import annotations

from typing import Any

import pytest

from gpt_trader.security.secrets_manager import SecretsManager


class TestEncryptionBootstrap:
    """Test encryption initialization scenarios."""

    def test_bootstrap_with_env_key(
        self, secrets_runtime_settings: Any, patched_require_fernet: None
    ) -> None:
        """Test bootstrap with encryption key from environment."""
        manager = SecretsManager(settings=secrets_runtime_settings)

        assert manager._cipher_suite is not None
        assert manager._vault_enabled is False  # Disabled by default in fixture

    def test_bootstrap_development_generates_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patched_require_fernet: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test bootstrap generates key in development when none provided."""
        # Remove encryption key from environment
        monkeypatch.setenv("GPT_TRADER_ENCRYPTION_KEY", "")

        from gpt_trader.config.runtime_settings import load_runtime_settings

        settings = load_runtime_settings()

        with caplog.at_level("WARNING"):
            manager = SecretsManager(settings=settings)

        assert manager._cipher_suite is not None
        assert any(
            "Generated new encryption key for development" in message for message in caplog.messages
        )

    def test_bootstrap_production_requires_key(
        self, monkeypatch: pytest.MonkeyPatch, patched_require_fernet: None
    ) -> None:
        """Test bootstrap requires key in production environment."""
        # Set production environment and remove key
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("GPT_TRADER_ENCRYPTION_KEY", "")

        from gpt_trader.config.runtime_settings import load_runtime_settings

        settings = load_runtime_settings()

        with pytest.raises(ValueError, match="ENCRYPTION_KEY must be set in production"):
            SecretsManager(settings=settings)

    def test_bootstrap_invalid_key_error(
        self, monkeypatch: pytest.MonkeyPatch, patched_require_fernet: None
    ) -> None:
        """Test bootstrap with invalid encryption key."""
        # Set invalid key
        monkeypatch.setenv("GPT_TRADER_ENCRYPTION_KEY", "invalid-key")

        from gpt_trader.config.runtime_settings import load_runtime_settings

        settings = load_runtime_settings()

        with pytest.raises(ValueError, match="Invalid encryption key"):
            SecretsManager(settings=settings)

    def test_bootstrap_key_types(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test bootstrap with different key types."""
        from gpt_trader.config.runtime_settings import load_runtime_settings

        # Test with string key
        monkeypatch.setenv("GPT_TRADER_ENCRYPTION_KEY", "Z9mB9nKp8sVqLyWsC5uE4oHj7gFdR2aL3xYiN6wTzQc=")
        settings = load_runtime_settings()

        manager = SecretsManager(settings=settings)
        assert manager._cipher_suite is not None

        # Test with bytes key represented as hex
        from cryptography.fernet import Fernet

        bytes_key = Fernet.generate_key().decode()
        monkeypatch.setenv("GPT_TRADER_ENCRYPTION_KEY", bytes_key)
        settings = load_runtime_settings()

        manager = SecretsManager(settings=settings)
        assert manager._cipher_suite is not None

    def test_deterministic_encryption(
        self,
        secrets_manager_with_fallback: Any,
        deterministic_fernet: Any,
        sample_secrets: dict[str, dict[str, Any]],
    ) -> None:
        """Test encryption is deterministic with fixed key."""
        manager = secrets_manager_with_fallback
        secret_path = "test/deterministic"
        secret_data = sample_secrets["brokers/coinbase"]

        # Store secret
        assert manager.store_secret(secret_path, secret_data)

        # Retrieve secret
        retrieved = manager.get_secret(secret_path)
        assert retrieved == secret_data

    def test_cipher_requirement_validation(
        self, secrets_runtime_settings: Any, patched_require_fernet: None
    ) -> None:
        """Test _require_cipher validation."""
        manager = SecretsManager(settings=secrets_runtime_settings)

        # Should return cipher suite
        cipher = manager._require_cipher()
        assert cipher is not None

        # Should raise if cipher not initialized (simulated)
        manager._cipher_suite = None
        with pytest.raises(RuntimeError, match="Encryption subsystem not initialised"):
            manager._require_cipher()

    def test_static_settings_preservation(
        self, secrets_runtime_settings: Any, patched_require_fernet: None
    ) -> None:
        """Test static settings are preserved during bootstrap."""
        manager = SecretsManager(settings=secrets_runtime_settings)

        assert manager._static_settings is True
        assert manager._settings is secrets_runtime_settings

    def test_dynamic_settings_loading(
        self, patched_require_fernet: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test dynamic settings loading when not provided."""
        # Set up environment
        monkeypatch.setenv("GPT_TRADER_ENCRYPTION_KEY", "Z9mB9nKp8sVqLyWsC5uE4oHj7gFdR2aL3xYiN6wTzQc=")

        manager = SecretsManager()  # No settings provided

        assert manager._static_settings is False
        assert manager._settings is not None
        assert manager._cipher_suite is not None

    def test_bootstrap_with_vault_enabled(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        hvac_stub: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test bootstrap with vault enabled."""
        # Enable vault
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        monkeypatch.setenv("VAULT_ADDR", "http://vault.local")

        # Create new settings with the updated environment
        from gpt_trader.config.runtime_settings import load_runtime_settings

        updated_settings = load_runtime_settings()

        manager = SecretsManager(vault_enabled=True, settings=updated_settings)

        assert manager._cipher_suite is not None
        assert manager._vault_enabled is True
        assert manager._vault_client is not None

    def test_bootstrap_vault_failure_fallback(
        self,
        secrets_runtime_settings: Any,
        patched_require_fernet: None,
        mock_vault_failure: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test fallback when vault initialization fails."""
        # Enable vault but mock failure
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        monkeypatch.setenv("VAULT_ADDR", "http://vault.local")

        # Mock vault to fail
        import sys

        monkeypatch.setitem(sys.modules, "hvac", mock_vault_failure)

        manager = SecretsManager(vault_enabled=True, settings=secrets_runtime_settings)

        assert manager._cipher_suite is not None
        assert manager._vault_enabled is False  # Should fallback to file storage

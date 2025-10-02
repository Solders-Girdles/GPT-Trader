"""Tests for SecretsManager - secure storage and encryption.

This module tests the SecretsManager's ability to securely store and retrieve
sensitive trading system credentials including API keys, database passwords,
and encryption keys. Tests verify:

- Encryption strength and key validation
- Secure file storage with proper permissions
- Thread-safe concurrent access
- Vault integration with proper fallbacks
- Secret rotation capabilities

Security Context:
    The SecretsManager is critical infrastructure for protecting production
    trading credentials. Failures here could expose API keys to unauthorized
    access, leading to account compromise, unauthorized trades, or financial loss.
"""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from cryptography.fernet import Fernet

from bot_v2.security.secrets_manager import SecretsManager, get_secrets_manager


@pytest.fixture
def encryption_key():
    """Generate valid encryption key for testing."""
    return Fernet.generate_key().decode()


@pytest.fixture
def temp_secrets_dir(tmp_path):
    """Provide temporary secrets directory."""
    secrets_dir = tmp_path / ".bot_v2" / "secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    return secrets_dir


@pytest.fixture
def secrets_manager(encryption_key, monkeypatch, tmp_path):
    """Create SecretsManager with test encryption key and temp directory."""
    monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", encryption_key)
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return SecretsManager(vault_enabled=False)


class TestEncryptionInitialization:
    """Test encryption key initialization and validation.

    These tests ensure that encryption keys are properly validated during
    initialization, preventing weak or malformed keys from being used.
    Proper key validation is essential for maintaining cryptographic security.
    """

    def test_valid_encryption_key_initialization(self, encryption_key, monkeypatch, tmp_path):
        """SecretsManager initializes with valid encryption key.

        Verifies that a properly formatted Fernet key is accepted and the
        cipher suite is initialized. This is the happy path for production
        environments where BOT_V2_ENCRYPTION_KEY is properly configured.
        """
        monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", encryption_key)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = SecretsManager(vault_enabled=False)

        assert manager._cipher_suite is not None

    def test_missing_key_in_production_raises_error(self, monkeypatch, tmp_path):
        """Missing encryption key in production environment raises ValueError.

        Critical security check: Production systems MUST fail fast if encryption
        keys are not configured. This prevents accidentally running with weak
        or auto-generated keys in production, which could lead to compromised
        credentials if the system is restarted with a different key.
        """
        monkeypatch.delenv("BOT_V2_ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with pytest.raises(ValueError, match="ENCRYPTION_KEY must be set in production"):
            SecretsManager(vault_enabled=False)

    def test_development_generates_key_when_missing(self, monkeypatch, tmp_path, caplog):
        """Development environment generates new key when missing.

        Development convenience: Auto-generates a key to enable local testing
        without manual configuration. The warning log ensures developers are
        aware this is happening and understand it's not suitable for production.
        """
        monkeypatch.delenv("BOT_V2_ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ENV", "development")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = SecretsManager(vault_enabled=False)

        assert manager._cipher_suite is not None
        assert "Generated new encryption key for development" in caplog.text

    def test_invalid_encryption_key_raises_error(self, monkeypatch, tmp_path):
        """Invalid encryption key format raises ValueError.

        Prevents using malformed keys that would fail during encryption/decryption.
        Catching this early (at initialization) is better than discovering it
        when trying to access credentials during a critical trading operation.
        """
        monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", "invalid-key-format")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with pytest.raises(ValueError, match="Invalid encryption key"):
            SecretsManager(vault_enabled=False)


class TestSecretStorage:
    """Test secret storage and retrieval operations.

    These tests verify the core functionality of encrypting, persisting, and
    retrieving secrets. Data integrity is critical - any corruption could
    render API keys unusable and halt trading operations.
    """

    def test_store_and_retrieve_secret(self, secrets_manager):
        """Store and retrieve secret maintains data integrity.

        Verifies the complete encryption/decryption round-trip works correctly.
        This is the fundamental operation that all secret management depends on.
        If this fails, no credentials can be safely stored or accessed.
        """
        secret = {"api_key": "test-key-123", "api_secret": "secret-456"}

        success = secrets_manager.store_secret("test/path", secret)
        assert success is True

        retrieved = secrets_manager.get_secret("test/path")
        assert retrieved == secret

    def test_store_secret_creates_encrypted_file(self, secrets_manager, tmp_path):
        """Store secret creates encrypted file on disk.

        Verifies that secrets are actually encrypted in storage, not just
        base64 encoded or obfuscated. We check that plaintext values are NOT
        present in the file contents, ensuring the encryption is real.

        Security: If secrets were stored in plaintext, they could be compromised
        by filesystem access, backups, or log files.
        """
        secret = {"key": "value"}
        secrets_dir = tmp_path / ".bot_v2" / "secrets"

        secrets_manager.store_secret("brokers/coinbase", secret)

        encrypted_file = secrets_dir / "brokers_coinbase.enc"
        assert encrypted_file.exists()

        # File should be encrypted (not plain JSON)
        content = encrypted_file.read_bytes()
        assert b"key" not in content  # Verify encryption
        assert b"value" not in content

    def test_retrieve_nonexistent_secret_returns_none(self, secrets_manager):
        """Retrieving non-existent secret returns None."""
        result = secrets_manager.get_secret("does/not/exist")
        assert result is None

    def test_secret_caching_avoids_disk_reads(self, secrets_manager, monkeypatch):
        """Cached secrets are returned without disk access.

        Performance optimization: Caching reduces disk I/O for frequently accessed
        credentials. This is important for high-frequency trading where every
        millisecond counts. We verify the cache is actually used by ensuring
        the file read method is never called for cached values.
        """
        secret = {"cached": "data"}
        secrets_manager.store_secret("cache/test", secret)

        # Mock file read to verify it's not called
        read_called = {"count": 0}
        original_load = secrets_manager._load_from_file

        def tracked_load(path):
            read_called["count"] += 1
            return original_load(path)

        monkeypatch.setattr(secrets_manager, "_load_from_file", tracked_load)

        # First call should read from cache (no disk access)
        retrieved = secrets_manager.get_secret("cache/test")
        assert retrieved == secret
        assert read_called["count"] == 0

    def test_clear_cache_forces_disk_read(self, secrets_manager):
        """Clear cache forces next get_secret to read from disk."""
        secret = {"key": "value"}
        secrets_manager.store_secret("test/cache", secret)

        # Verify cached
        assert secrets_manager._secrets_cache.get("test/cache") == secret

        # Clear cache
        secrets_manager.clear_cache()
        assert "test/cache" not in secrets_manager._secrets_cache

        # Should still retrieve from file
        retrieved = secrets_manager.get_secret("test/cache")
        assert retrieved == secret


class TestSecretDeletion:
    """Test secret deletion operations."""

    def test_delete_secret_removes_file_and_cache(self, secrets_manager, tmp_path):
        """Delete secret removes both file and cache entry."""
        secret = {"key": "value"}
        secrets_manager.store_secret("test/delete", secret)

        # Verify exists
        secrets_dir = tmp_path / ".bot_v2" / "secrets"
        encrypted_file = secrets_dir / "test_delete.enc"
        assert encrypted_file.exists()
        assert "test/delete" in secrets_manager._secrets_cache

        # Delete
        success = secrets_manager.delete_secret("test/delete")
        assert success is True

        # Verify removed
        assert not encrypted_file.exists()
        assert "test/delete" not in secrets_manager._secrets_cache

    def test_delete_nonexistent_secret_succeeds(self, secrets_manager):
        """Deleting non-existent secret returns True."""
        success = secrets_manager.delete_secret("does/not/exist")
        assert success is True


class TestKeyRotation:
    """Test encryption key rotation.

    Key rotation is a security best practice for limiting the impact of key
    compromise. These tests verify that secrets can be rotated while maintaining
    access to the underlying data and properly tracking rotation timestamps.
    """

    def test_rotate_key_updates_secret(self, secrets_manager):
        """Rotate key updates secret with new key and timestamp.

        Verifies that rotating a secret generates a new key value and records
        when the rotation occurred. This is critical for compliance requirements
        and security audits that mandate periodic credential rotation.
        """
        secret = {"_key": "old-key-123", "data": "sensitive"}
        secrets_manager.store_secret("test/rotate", secret)

        success = secrets_manager.rotate_key("test/rotate")
        assert success is True

        rotated = secrets_manager.get_secret("test/rotate")
        assert rotated["_key"] != "old-key-123"
        assert "_rotated_at" in rotated
        assert rotated["data"] == "sensitive"  # Other data preserved

    def test_rotate_key_on_nonexistent_secret_fails(self, secrets_manager):
        """Rotating key on non-existent secret returns False.

        Error handling: Attempting to rotate a secret that doesn't exist should
        fail gracefully with False return value, allowing callers to detect and
        handle missing secrets appropriately.
        """
        success = secrets_manager.rotate_key("does/not/exist")
        assert success is False

    def test_rotate_key_without_key_field_succeeds(self, secrets_manager):
        """Rotating key on secret without _key field succeeds without changes.

        Flexibility: Not all secrets have rotatable keys (some may be static
        configuration). Rotate operation should succeed gracefully for these,
        allowing uniform rotation commands across all secret types.
        """
        secret = {"data": "value"}
        secrets_manager.store_secret("test/no_key", secret)

        success = secrets_manager.rotate_key("test/no_key")
        assert success is True

        retrieved = secrets_manager.get_secret("test/no_key")
        assert retrieved == secret  # Unchanged


class TestThreadSafety:
    """Test thread safety of concurrent operations.

    Critical for production: Trading systems are inherently concurrent, with
    multiple threads handling orders, data feeds, and monitoring. If secret
    access isn't thread-safe, we risk race conditions that could corrupt data
    or cause deadlocks during live trading.
    """

    def test_concurrent_store_operations(self, secrets_manager):
        """Concurrent store operations are thread-safe.

        Simulates multiple threads writing different secrets simultaneously.
        Without proper locking, concurrent writes could corrupt the cache,
        lose data, or create inconsistent file states. All operations should
        complete successfully with correct data.
        """
        results = []

        def store_secret(path, value):
            secret = {"value": value}
            success = secrets_manager.store_secret(f"concurrent/{path}", secret)
            results.append(success)

        threads = [
            threading.Thread(target=store_secret, args=(f"key{i}", f"value{i}")) for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All operations should succeed
        assert all(results)
        assert len(results) == 10

        # All secrets should be retrievable
        for i in range(10):
            retrieved = secrets_manager.get_secret(f"concurrent/key{i}")
            assert retrieved == {"value": f"value{i}"}

    def test_concurrent_read_write_operations(self, secrets_manager):
        """Concurrent reads and writes maintain consistency.

        Real-world scenario: Multiple threads may be reading credentials while
        others are updating them (e.g., during key rotation). This test ensures
        readers never see partial/corrupted data and writers don't lose updates.
        All read results should be valid, complete secret dictionaries.
        """
        secrets_manager.store_secret("shared/key", {"value": "initial"})
        read_results = []

        def reader():
            for _ in range(5):
                result = secrets_manager.get_secret("shared/key")
                read_results.append(result)

        def writer(value):
            secrets_manager.store_secret("shared/key", {"value": value})

        readers = [threading.Thread(target=reader) for _ in range(3)]
        writers = [threading.Thread(target=writer, args=(f"v{i}",)) for i in range(3)]

        for t in readers + writers:
            t.start()
        for t in readers + writers:
            t.join()

        # All reads should return valid data (no corruption)
        for result in read_results:
            assert result is not None
            assert "value" in result


class TestVaultIntegration:
    """Test HashiCorp Vault integration.

    Vault is the enterprise solution for secret management but adds operational
    complexity. These tests verify that the system gracefully degrades to
    encrypted file storage when Vault is unavailable, ensuring trading can
    continue even if Vault infrastructure fails.
    """

    def test_vault_disabled_uses_file_storage(self, encryption_key, monkeypatch, tmp_path):
        """Vault disabled falls back to file storage.

        Verifies the explicit configuration path where Vault is intentionally
        disabled (e.g., for development or small deployments). System should
        work perfectly with file-based storage.
        """
        monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", encryption_key)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = SecretsManager(vault_enabled=False)

        assert manager._vault_enabled is False
        assert manager._vault_client is None

    def test_vault_missing_token_disables_vault(
        self, encryption_key, monkeypatch, tmp_path, caplog
    ):
        """Missing Vault token disables Vault and logs warning.

        Graceful degradation: If Vault is enabled but credentials are missing
        (misconfiguration), the system should fall back to file storage rather
        than crash. The warning ensures operators are aware of the fallback.
        This prevents trading halts due to Vault misconfiguration.
        """
        monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", encryption_key)
        monkeypatch.delenv("VAULT_TOKEN", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = SecretsManager(vault_enabled=True)

        assert manager._vault_enabled is False
        # May log either "hvac not installed" or "Vault token not found" depending on environment
        assert any(msg in caplog.text for msg in ["Vault token not found", "hvac not installed"])

    def test_vault_authentication_failure_falls_back(
        self, encryption_key, monkeypatch, tmp_path, caplog
    ):
        """Failed Vault authentication falls back to file storage.

        Note: Requires hvac installed. Mocking hvac import is complex, so we skip if unavailable.
        """
        pytest.importorskip("hvac")

        monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", encryption_key)
        monkeypatch.setenv("VAULT_TOKEN", "invalid-token")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Mock hvac Client to simulate auth failure
        with patch("hvac.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.is_authenticated.return_value = False
            mock_client_class.return_value = mock_client

            manager = SecretsManager(vault_enabled=True)

            assert manager._vault_enabled is False
            assert "Vault authentication failed" in caplog.text

    def test_vault_connection_error_falls_back(
        self, encryption_key, monkeypatch, tmp_path, caplog
    ):
        """Vault connection error during init falls back to file storage."""
        pytest.importorskip("hvac")

        monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", encryption_key)
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Mock hvac Client to raise connection error
        with patch("hvac.Client") as mock_client_class:
            mock_client_class.side_effect = Exception("Connection refused")

            manager = SecretsManager(vault_enabled=True)

            assert manager._vault_enabled is False
            assert "Vault initialization failed" in caplog.text

    def test_hvac_import_error_uses_file_storage(
        self, encryption_key, monkeypatch, tmp_path, caplog
    ):
        """Missing hvac package falls back to file storage gracefully."""
        monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", encryption_key)
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Simulate hvac not being installed
        with patch.dict("sys.modules", {"hvac": None}):
            manager = SecretsManager(vault_enabled=True)

            assert manager._vault_enabled is False
            assert "hvac not installed" in caplog.text

    def test_store_secret_vault_error_returns_false(
        self, encryption_key, monkeypatch, tmp_path, caplog
    ):
        """Store operation failing in Vault returns False and logs error."""
        pytest.importorskip("hvac")

        monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", encryption_key)
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Mock successful auth but failed store
        with patch("hvac.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.is_authenticated.return_value = True
            mock_client.secrets.kv.v2.create_or_update_secret.side_effect = Exception(
                "Vault write error"
            )
            mock_client_class.return_value = mock_client

            manager = SecretsManager(vault_enabled=True)
            assert manager._vault_enabled is True

            # Store should fail and return False
            result = manager.store_secret("test/key", {"value": "data"})
            assert result is False
            assert "Failed to store secret" in caplog.text

    def test_get_secret_vault_error_returns_none(
        self, encryption_key, monkeypatch, tmp_path, caplog
    ):
        """Get operation failing in Vault returns None and logs error."""
        pytest.importorskip("hvac")

        monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", encryption_key)
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        with patch("hvac.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.is_authenticated.return_value = True
            mock_client.secrets.kv.v2.read_secret_version.side_effect = Exception(
                "Vault read error"
            )
            mock_client_class.return_value = mock_client

            manager = SecretsManager(vault_enabled=True)

            # Get should fail and return None
            result = manager.get_secret("test/key")
            assert result is None
            assert "Failed to retrieve secret" in caplog.text

    def test_vault_fallback_continues_operations(
        self, encryption_key, monkeypatch, tmp_path
    ):
        """After vault failure, operations continue with file storage."""
        pytest.importorskip("hvac")

        monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", encryption_key)
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Vault fails during init
        with patch("hvac.Client") as mock_client_class:
            mock_client_class.side_effect = Exception("Vault unavailable")
            manager = SecretsManager(vault_enabled=True)

        # Should fall back to file storage
        assert manager._vault_enabled is False

        # Operations should still work via file storage
        test_secret = {"api_key": "test-key-123"}
        assert manager.store_secret("test/fallback", test_secret) is True

        retrieved = manager.get_secret("test/fallback")
        assert retrieved == test_secret


class TestSecretListing:
    """Test listing available secrets."""

    def test_list_secrets_from_file_storage(self, secrets_manager):
        """List secrets returns all stored secret paths."""
        secrets_manager.store_secret("brokers/coinbase", {"key": "1"})
        secrets_manager.store_secret("brokers/binance", {"key": "2"})
        secrets_manager.store_secret("api/keys/service", {"key": "3"})  # Underscore conversion

        paths = secrets_manager.list_secrets()

        # Path separators are converted to underscores in filenames, then back
        assert "brokers/coinbase" in paths
        assert "brokers/binance" in paths
        assert "api/keys/service" in paths
        assert len(paths) == 3

    def test_list_secrets_empty_directory(self, secrets_manager):
        """List secrets on empty directory returns empty list."""
        paths = secrets_manager.list_secrets()
        assert paths == []


class TestGlobalInstance:
    """Test global secrets manager instance."""

    def test_get_secrets_manager_returns_singleton(self, monkeypatch):
        """get_secrets_manager returns singleton instance."""
        monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", Fernet.generate_key().decode())
        monkeypatch.setenv("ENV", "development")

        # Reset global instance
        import bot_v2.security.secrets_manager as sm

        sm._secrets_manager = None

        manager1 = get_secrets_manager()
        manager2 = get_secrets_manager()

        assert manager1 is manager2

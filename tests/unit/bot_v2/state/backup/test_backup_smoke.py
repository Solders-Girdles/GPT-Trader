"""Smoke test for backup serialization pipeline.

Validates basic serialize → deserialize roundtrip to ensure
foundational components work before testing complex scenarios.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot_v2.state.backup.models import BackupConfig
from bot_v2.state.backup.serialization import BackupSerializer
from tests.unit.bot_v2.state.backup.conftest import (
    calculate_checksum,
    make_snapshot_payload,
)


class TestBackupSmokeTest:
    """Smoke tests for backup serialization pipeline."""

    def test_serialize_deserialize_roundtrip(
        self, backup_serializer: BackupSerializer, minimal_state_payload: dict
    ) -> None:
        """Serializes and deserializes simple payload without data loss.

        Critical smoke test: if this fails, entire backup system broken.
        """
        # Serialize
        serialized = backup_serializer.serialize_backup_data(minimal_state_payload)

        assert serialized is not None
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

        # Deserialize
        deserialized = json.loads(serialized.decode("utf-8"))

        # Verify roundtrip
        assert deserialized["timestamp"] == minimal_state_payload["timestamp"]
        assert deserialized["positions"] == minimal_state_payload["positions"]
        assert deserialized["metrics"] == minimal_state_payload["metrics"]

    def test_compression_reduces_size(
        self, backup_config: BackupConfig, sample_runtime_state: dict
    ) -> None:
        """Compression reduces payload size for repetitive data.

        Validates compression actually works.
        """
        # Enable compression
        backup_config.enable_compression = True
        serializer = BackupSerializer(backup_config)

        # Serialize with compression
        serialized = serializer.serialize_backup_data(sample_runtime_state)
        compressed, original_size, compressed_size = serializer.prepare_compressed_payload(
            serialized
        )

        assert compressed_size < original_size, "Compression should reduce size"
        assert compressed_size > 0, "Compressed data should not be empty"

    def test_checksum_calculation_works(
        self, backup_serializer: BackupSerializer, minimal_state_payload: dict
    ) -> None:
        """Checksum calculation produces valid hash.

        Critical for integrity verification.
        """
        serialized = backup_serializer.serialize_backup_data(minimal_state_payload)

        checksum = backup_serializer.calculate_checksum(serialized)

        assert checksum is not None
        assert len(checksum) == 64, "SHA-256 should produce 64-char hex"
        assert checksum.isalnum(), "Checksum should be alphanumeric hex"

    def test_checksum_verification_detects_changes(
        self, backup_serializer: BackupSerializer, minimal_state_payload: dict
    ) -> None:
        """Checksum verification detects data modifications.

        Essential for corruption detection.
        """
        serialized = backup_serializer.serialize_backup_data(minimal_state_payload)
        checksum = backup_serializer.calculate_checksum(serialized)

        # Modify data
        modified = serialized + b"CORRUPTED"

        # Verification should fail
        assert not backup_serializer.verify_checksum(modified, checksum)

    def test_checksum_verification_passes_unchanged_data(
        self, backup_serializer: BackupSerializer, minimal_state_payload: dict
    ) -> None:
        """Checksum verification passes for unchanged data.

        Ensures no false positives.
        """
        serialized = backup_serializer.serialize_backup_data(minimal_state_payload)
        checksum = backup_serializer.calculate_checksum(serialized)

        # Verification should succeed
        assert backup_serializer.verify_checksum(serialized, checksum)

    def test_handles_empty_payload(self, backup_serializer: BackupSerializer) -> None:
        """Handles empty payload without crashing.

        Edge case: backing up empty state.
        """
        empty_payload = {}

        serialized = backup_serializer.serialize_backup_data(empty_payload)

        assert serialized is not None
        assert len(serialized) > 0  # At least "{}"

        # Should deserialize
        deserialized = json.loads(serialized.decode("utf-8"))
        assert deserialized == {}

    def test_handles_large_payload(
        self, backup_serializer: BackupSerializer, temp_workspace: Path
    ) -> None:
        """Handles large payload without errors.

        Stress test: large state snapshot.
        """
        # Create large payload (100 positions)
        large_payload = make_snapshot_payload(
            positions={f"SYM_{i}": {"qty": 100 * i, "price": 150.0} for i in range(100)},
            metrics={f"metric_{i}": i * 1.5 for i in range(100)},
        )

        # Should serialize without error
        serialized = backup_serializer.serialize_backup_data(large_payload)

        assert len(serialized) > 1000, "Large payload should produce significant data"

        # Should calculate checksum
        checksum = backup_serializer.calculate_checksum(serialized)
        assert len(checksum) == 64

    def test_handles_special_characters_in_data(self, backup_serializer: BackupSerializer) -> None:
        """Handles special characters in payload without corruption.

        Ensures proper encoding/escaping.
        """
        payload_with_special_chars = {
            "field": "Value with special chars: \n\t\"quotes\" and 'apostrophes'",
            "unicode": "Unicode: 日本語, émojis: 🚀💰",
            "escape": "Backslash \\ and slash /",
        }

        serialized = backup_serializer.serialize_backup_data(payload_with_special_chars)
        deserialized = json.loads(serialized.decode("utf-8"))

        # All special chars preserved
        assert deserialized["field"] == payload_with_special_chars["field"]
        assert deserialized["unicode"] == payload_with_special_chars["unicode"]
        assert deserialized["escape"] == payload_with_special_chars["escape"]

    def test_nested_data_structures_preserved(self, backup_serializer: BackupSerializer) -> None:
        """Nested data structures preserved through serialization.

        Ensures deep data integrity.
        """
        nested_payload = {
            "level1": {
                "level2": {"level3": {"deep_value": 12345}, "another": [1, 2, 3]},
                "list_of_dicts": [{"a": 1}, {"b": 2}],
            }
        }

        serialized = backup_serializer.serialize_backup_data(nested_payload)
        deserialized = json.loads(serialized.decode("utf-8"))

        # Verify deep nesting preserved
        assert deserialized["level1"]["level2"]["level3"]["deep_value"] == 12345
        assert deserialized["level1"]["list_of_dicts"][0]["a"] == 1


class TestEncryptionBootstrap:
    """Tests for encryption key initialization and management."""

    def test_generates_new_encryption_key_when_none_exists(self, backup_config: BackupConfig) -> None:
        """Generates new encryption key when key file doesn't exist.

        Validates _init_encryption() creates new key.
        """
        from bot_v2.state.backup.serialization import ENCRYPTION_AVAILABLE

        if not ENCRYPTION_AVAILABLE:
            pytest.skip("Cryptography not available")

        backup_config.enable_encryption = True
        serializer = BackupSerializer(backup_config)

        # Key should be generated
        assert serializer._encryption_key is not None
        assert len(serializer._encryption_key) > 0

        # Key file should exist
        key_file = Path(backup_config.backup_dir) / ".encryption_key"
        assert key_file.exists()

    def test_loads_existing_encryption_key(self, backup_config: BackupConfig, temp_workspace: Path) -> None:
        """Loads existing encryption key from file.

        Validates _init_encryption() reuses existing key.
        """
        from bot_v2.state.backup.serialization import ENCRYPTION_AVAILABLE

        if not ENCRYPTION_AVAILABLE:
            pytest.skip("Cryptography not available")

        # Create existing key file
        from cryptography.fernet import Fernet

        existing_key = Fernet.generate_key()
        key_file = Path(backup_config.backup_dir) / ".encryption_key"
        key_file.parent.mkdir(parents=True, exist_ok=True)
        with open(key_file, "wb") as f:
            f.write(existing_key)

        # Initialize serializer - should load existing key
        backup_config.enable_encryption = True
        serializer = BackupSerializer(backup_config)

        assert serializer._encryption_key == existing_key

    def test_encryption_disabled_when_library_unavailable(
        self, backup_config: BackupConfig, monkeypatch
    ) -> None:
        """Handles missing cryptography library gracefully.

        Validates ENCRYPTION_AVAILABLE=False path.
        """
        # Mock ENCRYPTION_AVAILABLE as False
        import bot_v2.state.backup.serialization as serialization_module

        monkeypatch.setattr(serialization_module, "ENCRYPTION_AVAILABLE", False)

        backup_config.enable_encryption = True
        serializer = BackupSerializer(backup_config)

        # Should not have encryption key
        assert serializer._encryption_key is None

        # encrypt_payload should return data unchanged
        test_data = b"test data"
        encrypted, algorithm = serializer.encrypt_payload(test_data)
        assert encrypted == test_data
        assert algorithm is None

    def test_encryption_init_failure_handled_gracefully(
        self, backup_config: BackupConfig, monkeypatch
    ) -> None:
        """Handles encryption initialization errors gracefully.

        Validates exception handling in _init_encryption().
        """
        from bot_v2.state.backup.serialization import ENCRYPTION_AVAILABLE

        if not ENCRYPTION_AVAILABLE:
            pytest.skip("Cryptography not available")

        # Mock file operations to raise exception
        def failing_open(*args, **kwargs):
            raise PermissionError("Cannot write key file")

        monkeypatch.setattr("builtins.open", failing_open)

        backup_config.enable_encryption = True
        serializer = BackupSerializer(backup_config)

        # Should handle error - encryption key should be None
        assert serializer._encryption_key is None

    def test_encrypt_decrypt_roundtrip(self, backup_config: BackupConfig) -> None:
        """Encrypts and decrypts data successfully.

        Validates encryption/decryption works end-to-end.
        """
        from bot_v2.state.backup.serialization import ENCRYPTION_AVAILABLE

        if not ENCRYPTION_AVAILABLE:
            pytest.skip("Cryptography not available")

        backup_config.enable_encryption = True
        serializer = BackupSerializer(backup_config)

        original_data = b"sensitive backup data"

        # Encrypt
        encrypted, algorithm = serializer.encrypt_payload(original_data)
        assert encrypted != original_data
        assert algorithm == "Fernet"

        # Decrypt
        decrypted = serializer.decrypt_payload(encrypted)
        assert decrypted == original_data


class TestDecryptionErrorHandling:
    """Tests for decryption error propagation."""

    def test_decrypt_raises_on_invalid_encrypted_data(self, backup_config: BackupConfig) -> None:
        """Decryption raises exception on invalid encrypted data.

        Validates decrypt_payload() error propagation.
        """
        from bot_v2.state.backup.serialization import ENCRYPTION_AVAILABLE

        if not ENCRYPTION_AVAILABLE:
            pytest.skip("Cryptography not available")

        backup_config.enable_encryption = True
        serializer = BackupSerializer(backup_config)

        # Try to decrypt invalid data
        invalid_encrypted_data = b"this is not valid encrypted data"

        with pytest.raises(Exception):  # Fernet raises InvalidToken
            serializer.decrypt_payload(invalid_encrypted_data)

    def test_decrypt_returns_unchanged_when_no_key(self, backup_config: BackupConfig) -> None:
        """Returns data unchanged when no encryption key available.

        Validates decrypt_payload() without key.
        """
        backup_config.enable_encryption = False
        serializer = BackupSerializer(backup_config)

        test_data = b"unencrypted data"
        result = serializer.decrypt_payload(test_data)

        assert result == test_data

    def test_decrypt_returns_unchanged_when_encryption_unavailable(
        self, backup_config: BackupConfig, monkeypatch
    ) -> None:
        """Returns data unchanged when cryptography library unavailable.

        Validates ENCRYPTION_AVAILABLE=False path in decrypt.
        """
        import bot_v2.state.backup.serialization as serialization_module

        monkeypatch.setattr(serialization_module, "ENCRYPTION_AVAILABLE", False)

        backup_config.enable_encryption = True
        serializer = BackupSerializer(backup_config)

        test_data = b"data"
        result = serializer.decrypt_payload(test_data)

        assert result == test_data


class TestDecompressionFallback:
    """Tests for decompression failure handling."""

    def test_decompress_returns_original_on_invalid_compressed_data(
        self, backup_serializer: BackupSerializer
    ) -> None:
        """Returns original data when decompression fails.

        Validates decompress_payload() fallback for non-compressed data.
        """
        # Non-compressed data
        uncompressed_data = b"this is not compressed"

        # Should return original data when decompression fails
        result = backup_serializer.decompress_payload(uncompressed_data)

        assert result == uncompressed_data

    def test_decompress_handles_valid_compressed_data(self, backup_config: BackupConfig) -> None:
        """Decompresses valid compressed data successfully.

        Validates decompress_payload() success path.
        """
        from bot_v2.state.utils import compress_data

        backup_config.enable_compression = True
        serializer = BackupSerializer(backup_config)

        original_data = b"test data to compress" * 100
        compressed = compress_data(original_data)

        # Should decompress successfully
        decompressed = serializer.decompress_payload(compressed)
        assert decompressed == original_data

    def test_compression_decompression_roundtrip(self, backup_config: BackupConfig) -> None:
        """Compresses and decompresses data without loss.

        Full roundtrip validation.
        """
        backup_config.enable_compression = True
        serializer = BackupSerializer(backup_config)

        original = b"Repetitive data " * 50

        # Compress
        compressed, orig_size, comp_size = serializer.prepare_compressed_payload(original)
        assert comp_size < orig_size

        # Decompress
        decompressed = serializer.decompress_payload(compressed)
        assert decompressed == original

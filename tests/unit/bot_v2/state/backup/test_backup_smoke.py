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

    def test_handles_special_characters_in_data(
        self, backup_serializer: BackupSerializer
    ) -> None:
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

    def test_nested_data_structures_preserved(
        self, backup_serializer: BackupSerializer
    ) -> None:
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

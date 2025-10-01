"""Shared helper utilities for backup/recovery tests.

Provides test utilities for:
- Snapshot payload generation
- Corruption simulation
- Structure validation
- Checksum calculation
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from tests.unit.bot_v2.state.backup.conftest import (
    assert_snapshot_structure,
    calculate_checksum,
    calculate_payload_checksum,
    create_backup_metadata,
    make_snapshot_payload,
    mutate_payload_for_corruption,
)


class TestHelperUtilities:
    """Test the helper utilities themselves to ensure correctness."""

    def test_make_snapshot_payload_with_defaults(self) -> None:
        """make_snapshot_payload creates valid payload with defaults."""
        payload = make_snapshot_payload()

        assert "timestamp" in payload
        assert "positions" in payload
        assert "metrics" in payload
        assert payload["positions"] == {}
        assert payload["metrics"] == {}

    def test_make_snapshot_payload_with_custom_data(self) -> None:
        """make_snapshot_payload accepts custom positions and metrics."""
        positions = {"AAPL": {"qty": 100}}
        metrics = {"pnl": 1000.0}

        payload = make_snapshot_payload(positions=positions, metrics=metrics)

        assert payload["positions"] == positions
        assert payload["metrics"] == metrics

    def test_make_snapshot_payload_with_extra_fields(self) -> None:
        """make_snapshot_payload accepts additional kwargs."""
        payload = make_snapshot_payload(
            custom_field="test_value", circuit_breakers={"active": False}
        )

        assert payload["custom_field"] == "test_value"
        assert payload["circuit_breakers"] == {"active": False}

    def test_calculate_checksum_deterministic(self) -> None:
        """calculate_checksum produces deterministic results."""
        data = b"test data for checksum"

        checksum1 = calculate_checksum(data)
        checksum2 = calculate_checksum(data)

        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA-256 hex length

    def test_calculate_checksum_different_data(self) -> None:
        """calculate_checksum produces different hashes for different data."""
        data1 = b"data one"
        data2 = b"data two"

        checksum1 = calculate_checksum(data1)
        checksum2 = calculate_checksum(data2)

        assert checksum1 != checksum2

    def test_calculate_payload_checksum_canonical(self) -> None:
        """calculate_payload_checksum uses canonical JSON representation."""
        # Different key order, same data
        payload1 = {"b": 2, "a": 1}
        payload2 = {"a": 1, "b": 2}

        checksum1 = calculate_payload_checksum(payload1)
        checksum2 = calculate_payload_checksum(payload2)

        # Should be identical due to sorted keys
        assert checksum1 == checksum2

    def test_calculate_payload_checksum_handles_nested_data(self) -> None:
        """calculate_payload_checksum handles nested structures."""
        payload = {
            "nested": {"inner": {"value": 123}},
            "list": [1, 2, 3],
            "timestamp": datetime.utcnow(),
        }

        checksum = calculate_payload_checksum(payload)

        assert checksum is not None
        assert len(checksum) == 64

    def test_mutate_payload_for_corruption_changes_data(self) -> None:
        """mutate_payload_for_corruption produces different payload."""
        original = make_snapshot_payload(positions={"AAPL": {"qty": 100}})

        corrupted = mutate_payload_for_corruption(original)

        # Should differ from original
        assert corrupted != original
        # Original should be unchanged (shallow copy)
        assert "AAPL" in original["positions"]

    def test_mutate_payload_for_corruption_targets_timestamp(self) -> None:
        """mutate_payload_for_corruption corrupts timestamp field."""
        payload = {"timestamp": "2025-09-30T12:00:00Z", "data": "test"}

        corrupted = mutate_payload_for_corruption(payload)

        assert corrupted["timestamp"] == "CORRUPTED_TIMESTAMP"

    def test_mutate_payload_for_corruption_targets_positions(self) -> None:
        """mutate_payload_for_corruption corrupts positions data."""
        payload = make_snapshot_payload(positions={"AAPL": {"qty": 100}})

        corrupted = mutate_payload_for_corruption(payload)

        assert corrupted["positions"]["AAPL"] == "CORRUPTED_DATA"

    def test_assert_snapshot_structure_validates_directory(
        self, temp_workspace: Path
    ) -> None:
        """assert_snapshot_structure validates snapshot directory structure."""
        snapshot_dir = temp_workspace / "snapshots" / "test_snapshot"
        snapshot_dir.mkdir(parents=True)

        # Create required files
        (snapshot_dir / "data.backup").write_text("test data")
        (snapshot_dir / "metadata.meta").write_text('{"test": "meta"}')

        # Should not raise
        assert_snapshot_structure(snapshot_dir)

    def test_assert_snapshot_structure_fails_missing_directory(
        self, temp_workspace: Path
    ) -> None:
        """assert_snapshot_structure fails for missing directory."""
        missing_dir = temp_workspace / "nonexistent"

        with pytest.raises(AssertionError, match="Snapshot directory missing"):
            assert_snapshot_structure(missing_dir)

    def test_assert_snapshot_structure_fails_missing_data_file(
        self, temp_workspace: Path
    ) -> None:
        """assert_snapshot_structure fails when data file missing."""
        snapshot_dir = temp_workspace / "snapshots" / "incomplete"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "metadata.meta").write_text('{"test": "meta"}')

        with pytest.raises(AssertionError, match="No data file found"):
            assert_snapshot_structure(snapshot_dir)

    def test_assert_snapshot_structure_fails_missing_metadata(
        self, temp_workspace: Path
    ) -> None:
        """assert_snapshot_structure fails when metadata missing."""
        snapshot_dir = temp_workspace / "snapshots" / "incomplete"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "data.backup").write_text("test data")

        with pytest.raises(AssertionError, match="No metadata file found"):
            assert_snapshot_structure(snapshot_dir)

    def test_create_backup_metadata_with_defaults(self) -> None:
        """create_backup_metadata creates valid metadata with defaults."""
        metadata = create_backup_metadata()

        assert metadata.backup_id == "TEST_20250930_120000"
        assert metadata.size_bytes == 1024
        assert metadata.size_compressed == 512
        assert len(metadata.checksum) == 64

    def test_create_backup_metadata_with_overrides(self) -> None:
        """create_backup_metadata accepts field overrides."""
        from bot_v2.state.backup.models import BackupType

        metadata = create_backup_metadata(
            backup_id="CUSTOM_ID",
            backup_type=BackupType.INCREMENTAL,
            size_bytes=2048,
        )

        assert metadata.backup_id == "CUSTOM_ID"
        assert metadata.backup_type == BackupType.INCREMENTAL
        assert metadata.size_bytes == 2048

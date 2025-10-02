"""Shared fixtures for backup/recovery tests.

Provides reusable test fixtures for backup and recovery validation:
- Temporary workspace isolation
- Sample state payloads
- Mock state managers and event stores
- Deterministic checksum utilities
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.state.backup.models import (
    BackupConfig,
    BackupMetadata,
    BackupStatus,
    BackupType,
    StorageTier,
)
from bot_v2.state.backup.serialization import BackupSerializer


@pytest.fixture
def temp_workspace():
    """Create isolated temporary workspace for backup tests.

    Provides:
        - Unique temp directory per test
        - Automatic cleanup after test
        - Isolated filesystem root
    """
    with tempfile.TemporaryDirectory(prefix="backup_test_") as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "backups").mkdir(exist_ok=True)
        (workspace / "snapshots").mkdir(exist_ok=True)
        (workspace / "metadata").mkdir(exist_ok=True)
        yield workspace


@pytest.fixture
def backup_config(temp_workspace: Path) -> BackupConfig:
    """Create test backup configuration.

    Returns deterministic config for testing:
        - Disabled S3 (tests use local storage)
        - Compression enabled
        - Encryption disabled (simplifies test assertions)
    """
    return BackupConfig(
        backup_dir=str(temp_workspace / "backups"),
        local_storage_path=str(temp_workspace / "backups" / "local"),
        enable_encryption=False,  # Simplifies testing
        enable_compression=True,
        compression_level=6,
        verify_after_backup=False,  # Tests verify manually
        s3_bucket=None,  # Disable S3 for unit tests
        retention_full=90,
        retention_differential=30,
        retention_incremental=7,
    )


@pytest.fixture
def backup_serializer(backup_config: BackupConfig) -> BackupSerializer:
    """Create BackupSerializer instance for testing."""
    return BackupSerializer(backup_config)


@pytest.fixture
def sample_runtime_state() -> dict[str, Any]:
    """Sample runtime state payload for testing.

    Represents typical state snapshot with:
        - Positions
        - Metrics
        - Circuit breakers
        - Timestamps
    """
    return {
        "timestamp": "2025-09-30T12:00:00+00:00",
        "positions": {
            "AAPL": {
                "symbol": "AAPL",
                "quantity": Decimal("100.0"),
                "avg_price": Decimal("150.25"),
                "market_value": Decimal("15025.00"),
                "unrealized_pnl": Decimal("125.50"),
            },
            "GOOGL": {
                "symbol": "GOOGL",
                "quantity": Decimal("50.0"),
                "avg_price": Decimal("2800.00"),
                "market_value": Decimal("140000.00"),
                "unrealized_pnl": Decimal("-500.00"),
            },
        },
        "metrics": {
            "total_equity": Decimal("200000.00"),
            "cash_balance": Decimal("45000.00"),
            "portfolio_value": Decimal("155000.00"),
            "daily_pnl": Decimal("1250.75"),
            "sharpe_ratio": 1.85,
            "max_drawdown": -0.08,
        },
        "circuit_breakers": {
            "daily_loss_limit": False,
            "position_size_limit": False,
            "volatility_circuit": False,
        },
        "last_update": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def minimal_state_payload() -> dict[str, Any]:
    """Minimal state payload for basic tests."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "positions": {},
        "metrics": {},
    }


@pytest.fixture
def mock_event_store() -> Mock:
    """Mock EventStore for testing backup operations."""
    store = Mock()
    store.get_events_since = AsyncMock(return_value=[])
    store.append_event = AsyncMock(return_value=True)
    store.get_all_events = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_state_manager(sample_runtime_state: dict[str, Any]) -> Mock:
    """Mock state manager that provides snapshot capability."""
    manager = Mock()
    manager.create_snapshot = AsyncMock(return_value=sample_runtime_state)
    manager.restore_snapshot = AsyncMock(return_value=True)
    manager.get_state = AsyncMock(return_value=None)
    manager.set_state = AsyncMock(return_value=True)
    manager.batch_set_state = AsyncMock(side_effect=lambda items, **kwargs: len(items))
    manager.get_keys_by_pattern = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def mock_risk_state_manager() -> Mock:
    """Mock RiskStateManager for recovery tests."""
    manager = Mock()
    manager.get_runtime_snapshot = Mock(
        return_value={
            "positions": {},
            "risk_metrics": {},
            "circuit_breakers": {"active": []},
        }
    )
    manager.restore_from_snapshot = AsyncMock(return_value=True)
    return manager


def make_snapshot_payload(
    positions: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Factory function to create test snapshot payloads.

    Args:
        positions: Position data (optional)
        metrics: Metrics data (optional)
        **kwargs: Additional fields

    Returns:
        Complete snapshot payload with defaults
    """
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "positions": positions or {},
        "metrics": metrics or {},
    }
    payload.update(kwargs)
    return payload


def calculate_checksum(data: bytes) -> str:
    """Calculate deterministic SHA-256 checksum.

    Args:
        data: Raw bytes to hash

    Returns:
        Hex digest string
    """
    return hashlib.sha256(data).hexdigest()


def calculate_payload_checksum(payload: dict[str, Any]) -> str:
    """Calculate checksum for payload dictionary.

    Uses canonical JSON representation for consistency.

    Args:
        payload: Dictionary to hash

    Returns:
        Hex digest string
    """
    # Canonical JSON with sorted keys
    canonical_json = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode()).hexdigest()


def assert_snapshot_structure(snapshot_path: Path) -> None:
    """Assert snapshot has valid structure.

    Validates:
        - Directory exists
        - Contains data file
        - Contains metadata file

    Args:
        snapshot_path: Path to snapshot directory

    Raises:
        AssertionError: If structure is invalid
    """
    assert snapshot_path.exists(), f"Snapshot directory missing: {snapshot_path}"
    assert snapshot_path.is_dir(), f"Snapshot path is not a directory: {snapshot_path}"

    # Check for data file (various possible names)
    data_files = list(snapshot_path.glob("*.backup")) + list(snapshot_path.glob("*.dat"))
    assert len(data_files) > 0, f"No data file found in {snapshot_path}"

    # Check for metadata
    meta_files = list(snapshot_path.glob("*.meta")) + list(snapshot_path.glob("metadata.json"))
    assert len(meta_files) > 0, f"No metadata file found in {snapshot_path}"


def mutate_payload_for_corruption(payload: dict[str, Any]) -> dict[str, Any]:
    """Mutate payload to simulate corruption.

    Makes deterministic changes to test corruption detection.

    Args:
        payload: Original payload

    Returns:
        Corrupted payload (shallow copy with modifications)
    """
    corrupted = payload.copy()

    # Inject corruption
    if "timestamp" in corrupted:
        corrupted["timestamp"] = "CORRUPTED_TIMESTAMP"
    if "positions" in corrupted and corrupted["positions"]:
        # Corrupt first position
        first_key = next(iter(corrupted["positions"]))
        corrupted["positions"][first_key] = "CORRUPTED_DATA"

    return corrupted


def create_backup_metadata(
    backup_id: str = "TEST_20250930_120000",
    backup_type: BackupType = BackupType.FULL,
    checksum: str = "a" * 64,
    **kwargs: Any,
) -> BackupMetadata:
    """Factory function to create test BackupMetadata.

    Args:
        backup_id: Unique backup identifier
        backup_type: Type of backup
        checksum: SHA-256 checksum
        **kwargs: Override default fields

    Returns:
        BackupMetadata instance
    """
    defaults = {
        "backup_id": backup_id,
        "backup_type": backup_type,
        "timestamp": datetime.utcnow(),
        "size_bytes": 1024,
        "size_compressed": 512,
        "checksum": checksum,
        "encryption_key_id": None,
        "storage_tier": StorageTier.LOCAL,
        "retention_days": 90,
        "status": BackupStatus.COMPLETED,
    }
    defaults.update(kwargs)
    return BackupMetadata(**defaults)

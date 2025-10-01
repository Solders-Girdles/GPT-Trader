"""Shared fixtures for recovery tests.

Re-exports fixtures from backup tests for use in recovery validation.
"""

from tests.unit.bot_v2.state.backup.conftest import (
    backup_config,
    calculate_checksum,
    calculate_payload_checksum,
    make_snapshot_payload,
    mock_event_store,
    mock_risk_state_manager,
    mock_state_manager,
    mutate_payload_for_corruption,
    sample_runtime_state,
    temp_workspace,
)

__all__ = [
    "backup_config",
    "calculate_checksum",
    "calculate_payload_checksum",
    "make_snapshot_payload",
    "mock_event_store",
    "mock_risk_state_manager",
    "mock_state_manager",
    "mutate_payload_for_corruption",
    "sample_runtime_state",
    "temp_workspace",
]

"""Tests for recovery alerting utilities."""

from __future__ import annotations

import logging
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from bot_v2.state.recovery.alerting import RecoveryAlerter
from bot_v2.state.recovery.models import (
    FailureEvent,
    FailureType,
    RecoveryMode,
    RecoveryOperation,
    RecoveryStatus,
)


@pytest.fixture
def recovery_operation() -> RecoveryOperation:
    """Create a minimal recovery operation for alert tests."""
    failure_event = FailureEvent(
        failure_type=FailureType.DATA_CORRUPTION,
        timestamp=datetime.utcnow(),
        severity="critical",
        affected_components=["state"],
        error_message="Data corruption detected",
    )
    return RecoveryOperation(
        operation_id="op-123",
        failure_event=failure_event,
        recovery_mode=RecoveryMode.AUTOMATIC,
        status=RecoveryStatus.IN_PROGRESS,
        started_at=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_send_alert_persists_payload_and_logs_warning(
    mock_state_manager, recovery_operation, caplog
) -> None:
    """Alerts should be persisted and emit warning logs for normal priority."""
    caplog.set_level(logging.WARNING)
    alerter = RecoveryAlerter(mock_state_manager)

    await alerter.send_alert("System recovered", recovery_operation)

    await_args = mock_state_manager.set_state.await_args_list[0]
    key, payload = await_args.args
    assert key.startswith("alert:op-123:")
    assert payload["message"] == "System recovered"
    assert payload["priority"] == "normal"
    assert payload["operation_id"] == "op-123"
    assert "Alert: System recovered" in caplog.text


@pytest.mark.asyncio
async def test_send_alert_high_priority_uses_critical_log(
    mock_state_manager, recovery_operation, caplog
) -> None:
    """High priority alerts should raise visibility via critical logging."""
    caplog.set_level(logging.CRITICAL)
    alerter = RecoveryAlerter(mock_state_manager)

    await alerter.send_alert("Escalate immediately", recovery_operation, priority="high")

    critical_messages = [
        record.message for record in caplog.records if record.levelno == logging.CRITICAL
    ]
    assert any("ALERT: Escalate immediately" in message for message in critical_messages)


@pytest.mark.asyncio
async def test_send_alert_handles_state_write_errors(
    mock_state_manager, recovery_operation, caplog
) -> None:
    """Failure to persist the alert should be captured without raising."""
    mock_state_manager.set_state.side_effect = RuntimeError("redis unreachable")
    caplog.set_level(logging.ERROR)
    alerter = RecoveryAlerter(mock_state_manager)

    await alerter.send_alert("Persistence failure", recovery_operation)

    assert "Failed to send alert: redis unreachable" in caplog.text


@pytest.mark.asyncio
async def test_escalate_recovery_creates_manual_recovery_record(
    mock_state_manager, recovery_operation, monkeypatch, caplog
) -> None:
    """Escalation should emit a manual recovery entry and critical checklist log."""
    caplog.set_level(logging.CRITICAL)
    alerter = RecoveryAlerter(mock_state_manager)
    send_alert_stub = AsyncMock()
    monkeypatch.setattr(alerter, "send_alert", send_alert_stub)

    await alerter.escalate_recovery(recovery_operation)

    send_alert_stub.assert_awaited_once()
    manual_args = mock_state_manager.set_state.await_args_list[-1].args
    key, value = manual_args
    assert key == "system:manual_recovery_required"
    assert value["operation_id"] == "op-123"
    assert "Manual recovery required" in caplog.text


@pytest.mark.asyncio
async def test_escalate_recovery_logs_errors(
    mock_state_manager, recovery_operation, caplog
) -> None:
    """Errors during escalation should be logged rather than raised."""
    mock_state_manager.set_state.side_effect = RuntimeError("write failed")
    caplog.set_level(logging.ERROR)
    alerter = RecoveryAlerter(mock_state_manager)

    await alerter.escalate_recovery(recovery_operation)

    assert "Recovery escalation failed: write failed" in caplog.text


class TestManualRecoveryChecklist:
    """Unit tests for manual recovery checklist generation."""

    def test_known_failure_type_has_dedicated_runbook(
        self, mock_state_manager, recovery_operation
    ) -> None:
        alerter = RecoveryAlerter(mock_state_manager)
        checklist = alerter.generate_manual_recovery_checklist(recovery_operation)
        assert "Stop all trading operations immediately" in checklist

    def test_unknown_failure_type_uses_fallback(
        self, mock_state_manager, recovery_operation
    ) -> None:
        recovery_operation.failure_event.failure_type = FailureType.REDIS_DOWN
        alerter = RecoveryAlerter(mock_state_manager)
        checklist = alerter.generate_manual_recovery_checklist(recovery_operation)
        assert "Assess situation" in checklist

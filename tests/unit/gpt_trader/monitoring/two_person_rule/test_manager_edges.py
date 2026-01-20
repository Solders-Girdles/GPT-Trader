"""Edge-case tests for TwoPersonRule manager."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import gpt_trader.monitoring.two_person_rule.manager as manager_module
from gpt_trader.monitoring.two_person_rule.manager import (
    TwoPersonRule,
    log_config_delta,
)
from gpt_trader.monitoring.two_person_rule.models import (
    ApprovalStatus,
    ChangeType,
    ConfigChange,
)


def test_create_request_sets_pending_and_uses_window() -> None:
    rule = TwoPersonRule(approval_window=timedelta(hours=2))
    requester = SimpleNamespace(id="user-1", username="user1")
    change = ConfigChange(ChangeType.LEVERAGE, "max_leverage", 1, 2)

    request = rule.create_request(requester=requester, changes=[change])

    assert request.status == ApprovalStatus.PENDING
    assert request.expires_at - request.created_at == timedelta(hours=2)


def test_approve_request_rejects_same_user_and_allows_different_user() -> None:
    rule = TwoPersonRule()
    requester = SimpleNamespace(id="user-1", username="user1")
    approver = SimpleNamespace(id="user-2", username="user2")
    change = ConfigChange(ChangeType.LEVERAGE, "max_leverage", 1, 2)

    request = rule.create_request(requester=requester, changes=[change])

    success, error = rule.approve_request(request_id=request.request_id, approver=requester)
    assert success is False
    assert error == "Requester cannot approve their own request"

    success, error = rule.approve_request(request_id=request.request_id, approver=approver)
    assert success is True
    assert error is None
    assert request.status == ApprovalStatus.APPROVED


def test_mark_applied_only_succeeds_from_approved() -> None:
    rule = TwoPersonRule()
    requester = SimpleNamespace(id="user-1", username="user1")
    approver = SimpleNamespace(id="user-2", username="user2")
    change = ConfigChange(ChangeType.LEVERAGE, "max_leverage", 1, 2)

    request = rule.create_request(requester=requester, changes=[change])
    success, _ = rule.mark_applied(request.request_id)
    assert success is False
    assert request.status == ApprovalStatus.PENDING

    rule.approve_request(request_id=request.request_id, approver=approver)
    success, _ = rule.mark_applied(request.request_id)
    assert success is True
    assert request.status == ApprovalStatus.APPLIED


def test_get_pending_requests_marks_expired() -> None:
    rule = TwoPersonRule()
    requester = SimpleNamespace(id="user-1", username="user1")
    change = ConfigChange(ChangeType.LEVERAGE, "max_leverage", 1, 2)

    expired = rule.create_request(requester=requester, changes=[change])
    expired.expires_at = datetime.now(UTC) - timedelta(seconds=1)

    active = rule.create_request(requester=requester, changes=[change])
    active.expires_at = datetime.now(UTC) + timedelta(hours=1)

    pending = rule.get_pending_requests()

    assert expired.status == ApprovalStatus.EXPIRED
    assert active in pending
    assert expired not in pending


def test_requires_approval_filters_critical_fields() -> None:
    rule = TwoPersonRule()
    changes = {
        "max_leverage": 5,
        "daily_loss_limit": 200,
        "some_other_field": "value",
    }

    required = rule.requires_approval(changes)

    assert set(required) == {"max_leverage", "daily_loss_limit"}


def test_log_config_delta_emits_metric_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    event_store = MagicMock()
    changes = {"max_leverage": (3, 5)}

    mock_emit = MagicMock()
    monkeypatch.setattr(manager_module, "emit_metric", mock_emit)

    log_config_delta(
        change_type="risk_limit_update",
        changes=changes,
        user_id="admin-1",
        metadata={"ticket": "RISK-1234"},
        event_store=event_store,
        bot_id="config_guardian",
    )

    mock_emit.assert_called_once()
    assert mock_emit.call_args[0][0] is event_store
    assert mock_emit.call_args[0][1] == "config_guardian"
    payload = mock_emit.call_args[0][2]
    assert payload["event_type"] == "config_delta"
    assert payload["change_type"] == "risk_limit_update"
    assert payload["changes"] == changes
    assert payload["user_id"] == "admin-1"

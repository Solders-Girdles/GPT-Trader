"""Edge-case tests for two-person rule models."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

sys.modules.setdefault(
    "gpt_trader.monitoring.two_person_rule.manager",
    SimpleNamespace(TwoPersonRule=object()),
)

from gpt_trader.monitoring.two_person_rule.models import (  # noqa: E402
    ApprovalRequest,
    ApprovalStatus,
    ChangeType,
    ConfigChange,
)


class _Dummy:
    def __str__(self) -> str:
        return "dummy-object"


def test_config_change_stringify_serializable_and_fallback() -> None:
    assert ConfigChange._stringify({"key": "value"}) == json.dumps({"key": "value"})
    assert ConfigChange._stringify(_Dummy()) == "dummy-object"


def test_approval_request_is_expired_only_when_pending() -> None:
    past = datetime.now(UTC) - timedelta(seconds=1)
    future = datetime.now(UTC) + timedelta(seconds=1)
    change = ConfigChange(ChangeType.LEVERAGE, "max_leverage", 1, 2)

    pending = ApprovalRequest(
        request_id="req-1",
        requester_id="user-1",
        requester_name="User",
        changes=[change],
        status=ApprovalStatus.PENDING,
        created_at=past,
        expires_at=past,
    )
    approved = ApprovalRequest(
        request_id="req-2",
        requester_id="user-1",
        requester_name="User",
        changes=[change],
        status=ApprovalStatus.APPROVED,
        created_at=past,
        expires_at=past,
    )
    not_expired = ApprovalRequest(
        request_id="req-3",
        requester_id="user-1",
        requester_name="User",
        changes=[change],
        status=ApprovalStatus.PENDING,
        created_at=past,
        expires_at=future,
    )

    assert pending.is_expired is True
    assert approved.is_expired is False
    assert not_expired.is_expired is False


def test_approval_request_mark_expired_respects_status() -> None:
    past = datetime.now(UTC) - timedelta(seconds=1)
    future = datetime.now(UTC) + timedelta(seconds=1)
    change = ConfigChange(ChangeType.LEVERAGE, "max_leverage", 1, 2)

    pending = ApprovalRequest(
        request_id="req-1",
        requester_id="user-1",
        requester_name="User",
        changes=[change],
        status=ApprovalStatus.PENDING,
        created_at=past,
        expires_at=past,
    )
    pending.mark_expired()
    assert pending.status == ApprovalStatus.EXPIRED

    approved = ApprovalRequest(
        request_id="req-2",
        requester_id="user-1",
        requester_name="User",
        changes=[change],
        status=ApprovalStatus.APPROVED,
        created_at=past,
        expires_at=past,
    )
    approved.mark_expired()
    assert approved.status == ApprovalStatus.APPROVED

    not_expired = ApprovalRequest(
        request_id="req-3",
        requester_id="user-1",
        requester_name="User",
        changes=[change],
        status=ApprovalStatus.PENDING,
        created_at=past,
        expires_at=future,
    )
    not_expired.mark_expired()
    assert not_expired.status == ApprovalStatus.PENDING


def test_approval_request_requires_approval_only_when_pending() -> None:
    change = ConfigChange(ChangeType.LEVERAGE, "max_leverage", 1, 2)
    pending = ApprovalRequest(
        request_id="req-1",
        requester_id="user-1",
        requester_name="User",
        changes=[change],
        status=ApprovalStatus.PENDING,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC),
    )
    approved = ApprovalRequest(
        request_id="req-2",
        requester_id="user-1",
        requester_name="User",
        changes=[change],
        status=ApprovalStatus.APPROVED,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC),
    )

    assert pending.requires_approval is True
    assert approved.requires_approval is False


def test_approval_request_to_dict_includes_fields_and_nones() -> None:
    created_at = datetime(2024, 1, 1, tzinfo=UTC)
    expires_at = datetime(2024, 1, 2, tzinfo=UTC)
    change = ConfigChange(ChangeType.LEVERAGE, "max_leverage", 1, 2)
    request = ApprovalRequest(
        request_id="req-1",
        requester_id="user-1",
        requester_name="User",
        changes=[change],
        status=ApprovalStatus.PENDING,
        created_at=created_at,
        expires_at=expires_at,
    )

    payload = request.to_dict()

    assert payload["status"] == "pending"
    assert payload["created_at"] == created_at.isoformat()
    assert payload["expires_at"] == expires_at.isoformat()
    assert payload["approver_id"] is None
    assert payload["approver_name"] is None
    assert payload["approved_at"] is None
    assert payload["rejection_reason"] is None
    assert payload["applied_at"] is None
    assert payload["changes"][0]["change_type"] == "leverage"
    assert payload["changes"][0]["field_name"] == "max_leverage"

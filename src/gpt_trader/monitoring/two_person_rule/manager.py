"""Two-person rule manager for configuration changes."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.telemetry import emit_metric

from .models import ApprovalRequest, ApprovalStatus, ConfigChange

logger = get_logger(__name__, component="two_person_rule")

DEFAULT_APPROVAL_WINDOW = timedelta(hours=24)
DEFAULT_BOT_ID = "config_guardian"

CRITICAL_FIELDS = {
    "max_leverage",
    "max_position_size",
    "daily_loss_limit",
    "liquidation_buffer",
    "circuit_breaker_threshold",
    "kill_switch",
    "reduce_only_mode",
    "symbols",
    "profile",
    "per_symbol_leverage",
}


def _get_user_attr(user: Any, attr: str) -> Any:
    if isinstance(user, dict):
        return user.get(attr)
    return getattr(user, attr, None)


def _normalize_user(user: Any) -> tuple[str, str]:
    if isinstance(user, str):
        return user, user
    user_id = _get_user_attr(user, "id") or _get_user_attr(user, "user_id")
    user_name = _get_user_attr(user, "username") or _get_user_attr(user, "name")
    user_email = _get_user_attr(user, "email")
    if user_name:
        return str(user_id or user_name), str(user_name)
    if user_email:
        return str(user_id or user_email), str(user_email)
    if user_id:
        return str(user_id), str(user_id)
    return str(user), str(user)


class TwoPersonRule:
    """Manages two-person approval workflows for configuration changes."""

    def __init__(
        self,
        *,
        approval_window: timedelta = DEFAULT_APPROVAL_WINDOW,
        critical_fields: Iterable[str] | None = None,
        event_store: Any | None = None,
        bot_id: str = DEFAULT_BOT_ID,
    ) -> None:
        self._approval_window = approval_window
        self._critical_fields = set(critical_fields or CRITICAL_FIELDS)
        self._event_store = event_store
        self._bot_id = bot_id
        self._requests: dict[str, ApprovalRequest] = {}

    def create_request(
        self,
        *,
        requester: Any,
        changes: list[ConfigChange],
        metadata: dict[str, Any] | None = None,
        approval_window: timedelta | None = None,
    ) -> ApprovalRequest:
        now = datetime.now(UTC)
        window = approval_window or self._approval_window
        request_id = f"APR-{uuid.uuid4().hex}"
        requester_id, requester_name = _normalize_user(requester)
        request = ApprovalRequest(
            request_id=request_id,
            requester_id=requester_id,
            requester_name=requester_name,
            changes=changes,
            status=ApprovalStatus.PENDING,
            created_at=now,
            expires_at=now + window,
            metadata=metadata or {},
        )
        self._requests[request_id] = request
        self._emit_request_event("approval_request_created", request)
        return request

    def approve_request(self, *, request_id: str, approver: Any) -> tuple[bool, str | None]:
        request = self._requests.get(request_id)
        if request is None:
            return False, "Request not found"
        if request.status != ApprovalStatus.PENDING:
            return False, "Request not pending"
        if request.is_expired:
            self._expire_request(request)
            return False, "Request expired"

        approver_id, approver_name = _normalize_user(approver)
        if request.requester_id == approver_id:
            return False, "Requester cannot approve their own request"

        request.status = ApprovalStatus.APPROVED
        request.approver_id = approver_id
        request.approver_name = approver_name
        request.approved_at = datetime.now(UTC)
        self._emit_request_event("approval_request_approved", request)
        return True, None

    def reject_request(
        self, *, request_id: str, reviewer: Any, reason: str | None = None
    ) -> tuple[bool, str | None]:
        request = self._requests.get(request_id)
        if request is None:
            return False, "Request not found"
        if request.status != ApprovalStatus.PENDING:
            return False, "Request not pending"
        if request.is_expired:
            self._expire_request(request)
            return False, "Request expired"

        reviewer_id, reviewer_name = _normalize_user(reviewer)
        request.status = ApprovalStatus.REJECTED
        request.approver_id = reviewer_id
        request.approver_name = reviewer_name
        request.rejection_reason = reason
        request.approved_at = datetime.now(UTC)
        self._emit_request_event("approval_request_rejected", request)
        return True, None

    def mark_applied(self, request_id: str) -> tuple[bool, str | None]:
        request = self._requests.get(request_id)
        if request is None:
            return False, "Request not found"
        if request.status != ApprovalStatus.APPROVED:
            return False, "Request not approved"

        request.status = ApprovalStatus.APPLIED
        request.applied_at = datetime.now(UTC)
        self._emit_request_event("approval_request_applied", request)
        return True, None

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        return self._requests.get(request_id)

    def get_pending_requests(self) -> list[ApprovalRequest]:
        pending: list[ApprovalRequest] = []
        for request in self._requests.values():
            if request.status == ApprovalStatus.PENDING and request.is_expired:
                self._expire_request(request)
            if request.status == ApprovalStatus.PENDING:
                pending.append(request)
        return pending

    def requires_approval(self, changes: dict[str, Any]) -> list[str]:
        return [field for field in changes if field in self._critical_fields]

    # ------------------------------------------------------------------
    def _expire_request(self, request: ApprovalRequest) -> None:
        request.status = ApprovalStatus.EXPIRED
        self._emit_request_event("approval_request_expired", request)

    def _emit_request_event(self, event_type: str, request: ApprovalRequest) -> None:
        emit_metric(
            self._event_store,
            self._bot_id,
            {
                "event_type": event_type,
                "request_id": request.request_id,
                "status": request.status.value,
                "requester_id": request.requester_id,
                "approver_id": request.approver_id,
            },
            logger=logger,
        )


_two_person_rule: TwoPersonRule | None = None


def get_two_person_rule(
    *,
    event_store: Any | None = None,
    bot_id: str = DEFAULT_BOT_ID,
    approval_window: timedelta = DEFAULT_APPROVAL_WINDOW,
    critical_fields: Iterable[str] | None = None,
) -> TwoPersonRule:
    global _two_person_rule
    if _two_person_rule is None:
        _two_person_rule = TwoPersonRule(
            approval_window=approval_window,
            critical_fields=critical_fields,
            event_store=event_store,
            bot_id=bot_id,
        )
    elif event_store is not None and _two_person_rule._event_store is None:
        _two_person_rule._event_store = event_store
    return _two_person_rule


def create_approval_request(
    *,
    requester: Any,
    changes: list[ConfigChange],
    metadata: dict[str, Any] | None = None,
    approval_window: timedelta | None = None,
) -> ApprovalRequest:
    return get_two_person_rule().create_request(
        requester=requester,
        changes=changes,
        metadata=metadata,
        approval_window=approval_window,
    )


def approve_request(*, request_id: str, approver: Any) -> tuple[bool, str | None]:
    return get_two_person_rule().approve_request(request_id=request_id, approver=approver)


def reject_request(
    *, request_id: str, reviewer: Any, reason: str | None = None
) -> tuple[bool, str | None]:
    return get_two_person_rule().reject_request(
        request_id=request_id, reviewer=reviewer, reason=reason
    )


def mark_applied(request_id: str) -> tuple[bool, str | None]:
    return get_two_person_rule().mark_applied(request_id)


def log_config_delta(
    *,
    change_type: str,
    changes: dict[str, tuple[Any, Any]],
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    event_store: Any | None = None,
    bot_id: str | None = None,
) -> None:
    rule = get_two_person_rule()
    store = event_store or rule._event_store
    if store is None:
        return
    emit_metric(
        store,
        bot_id or rule._bot_id,
        {
            "event_type": "config_delta",
            "change_type": change_type,
            "changes": changes,
            "user_id": user_id,
            "metadata": metadata or {},
        },
        logger=logger,
    )


__all__ = [
    "TwoPersonRule",
    "get_two_person_rule",
    "create_approval_request",
    "approve_request",
    "reject_request",
    "mark_applied",
    "log_config_delta",
]

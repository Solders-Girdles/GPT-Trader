"""Two-person rule manager implementation."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

from bot_v2.persistence.event_store import EventStore
from bot_v2.security.auth_handler import User
from bot_v2.utilities.logging_patterns import get_logger

from .models import ApprovalRequest, ApprovalStatus, ChangeType, ConfigChange

logger = get_logger(__name__, component="two_person_rule")


class TwoPersonRule:
    """Enforces the two-person rule for critical configuration changes."""

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

    def __init__(
        self,
        event_store: EventStore | None = None,
        approval_timeout_hours: int = 24,
        bot_id: str = "config_guardian",
    ) -> None:
        self._event_store = event_store or EventStore()
        self._approval_timeout_hours = approval_timeout_hours
        self._bot_id = bot_id
        self._lock = threading.RLock()
        self._pending_requests: dict[str, ApprovalRequest] = {}
        self._request_history: list[ApprovalRequest] = []
        self._max_history_size = 1000

    # ------------------------------------------------------------------ #
    # Lifecycle                                                         #
    # ------------------------------------------------------------------ #
    def create_approval_request(
        self,
        requester: User,
        changes: list[ConfigChange],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        with self._lock:
            now = datetime.now(UTC)
            expires_at = now + timedelta(hours=self._approval_timeout_hours)
            request_id = self._generate_request_id(requester.id, changes, now)

            request = ApprovalRequest(
                request_id=request_id,
                requester_id=requester.id,
                requester_name=requester.username,
                changes=changes,
                status=ApprovalStatus.PENDING,
                created_at=now,
                expires_at=expires_at,
                metadata=metadata or {},
            )

            self._pending_requests[request_id] = request
            self._add_to_history(request)
            self._log_approval_event("approval_request_created", request)

            logger.info(
                "Created approval request %s by %s",
                request_id,
                requester.username,
                extra={"operation": "create_approval_request", "change_count": len(changes)},
            )

            return request

    def approve_request(self, request_id: str, approver: User) -> tuple[bool, str | None]:
        with self._lock:
            request = self._pending_requests.get(request_id)

            if not request:
                return False, f"Request {request_id} not found"

            if request.status != ApprovalStatus.PENDING:
                return False, f"Request is in {request.status.value} status, cannot approve"

            if request.is_expired:
                request.status = ApprovalStatus.EXPIRED
                self._log_approval_event("approval_request_expired", request)
                self._add_to_history(request)
                self._pending_requests.pop(request_id, None)
                return False, "Request has expired"

            if approver.id == request.requester_id:
                logger.warning(
                    "Approval attempt rejected - same user",
                    operation="approve_request",
                    request_id=request_id,
                    user=approver.username,
                )
                return False, "Approver cannot be the same as requester (two-person rule)"

            now = datetime.now(UTC)
            request.status = ApprovalStatus.APPROVED
            request.approver_id = approver.id
            request.approver_name = approver.username
            request.approved_at = now

            self._log_approval_event("approval_request_approved", request)
            logger.info(
                "Approved request %s by %s",
                request_id,
                approver.username,
                extra={"operation": "approve_request", "change_count": len(request.changes)},
            )

            return True, None

    def reject_request(
        self,
        request_id: str,
        reviewer: User,
        reason: str,
    ) -> tuple[bool, str | None]:
        with self._lock:
            request = self._pending_requests.get(request_id)

            if not request:
                return False, f"Request {request_id} not found"

            if request.status != ApprovalStatus.PENDING:
                return False, f"Request is in {request.status.value} status, cannot reject"

            request.status = ApprovalStatus.REJECTED
            request.approver_id = reviewer.id
            request.approver_name = reviewer.username
            request.rejection_reason = reason

            self._pending_requests.pop(request_id, None)
            self._add_to_history(request)
            self._log_approval_event("approval_request_rejected", request)

            logger.info(
                "Rejected request %s by %s",
                request_id,
                reviewer.username,
                extra={"operation": "reject_request", "reason": reason},
            )

            return True, None

    def mark_applied(self, request_id: str) -> tuple[bool, str | None]:
        with self._lock:
            request = self._pending_requests.get(request_id)

            if not request:
                return False, f"Request {request_id} not found"

            if request.status != ApprovalStatus.APPROVED:
                return False, "Request must be approved before marking as applied"

            request.status = ApprovalStatus.APPLIED
            request.applied_at = datetime.now(UTC)
            self._pending_requests.pop(request_id, None)
            self._add_to_history(request)
            self._log_approval_event("approval_request_applied", request)

            logger.info(
                "Marked request %s as applied",
                request_id,
                extra={"operation": "mark_applied"},
            )

            return True, None

    def get_pending_requests(self) -> list[ApprovalRequest]:
        with self._lock:
            self._cleanup_expired_requests()
            return [req for req in self._pending_requests.values() if req.status == ApprovalStatus.PENDING]

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        with self._lock:
            return self._pending_requests.get(request_id)

    def get_request_history(self, limit: int = 100) -> list[ApprovalRequest]:
        with self._lock:
            return self._request_history[-limit:]

    def requires_approval(self, config_changes: dict[str, Any]) -> list[str]:
        return [field for field in config_changes if field in self.CRITICAL_FIELDS]

    def log_config_delta(
        self,
        change_type: str,
        changes: dict[str, tuple[Any, Any]],
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event_data = {
            "event_type": "config_delta",
            "change_type": change_type,
            "changes": {
                field: {"old": str(old), "new": str(new)} for field, (old, new) in changes.items()
            },
            "user_id": user_id,
            "metadata": metadata or {},
        }

        try:
            self._event_store.append_metric(self._bot_id, event_data)
        except Exception:
            logger.debug("Failed to log config delta", exc_info=True)

        logger.info(
            "Logged config delta: %s",
            change_type,
            extra={"operation": "log_config_delta", "field_count": len(changes)},
        )

    def expire_pending_requests(self) -> list[ApprovalRequest]:
        expired: list[ApprovalRequest] = []
        with self._lock:
            for request_id, request in list(self._pending_requests.items()):
                if request.is_expired:
                    request.status = ApprovalStatus.EXPIRED
                    expired.append(request)
                    self._pending_requests.pop(request_id, None)
                    self._add_to_history(request)
                    self._log_approval_event("approval_request_expired", request)

                    logger.info(
                        "Approval request %s expired",
                        request_id,
                        extra={"operation": "expire_request"},
                    )
        return expired

    # ------------------------------------------------------------------ #
    # Internals                                                         #
    # ------------------------------------------------------------------ #
    def _cleanup_expired_requests(self) -> None:
        expired = self.expire_pending_requests()
        if expired:
            logger.debug("Expired %d pending requests", len(expired))

    def _add_to_history(self, request: ApprovalRequest) -> None:
        self._request_history.append(request)
        if len(self._request_history) > self._max_history_size:
            self._request_history = self._request_history[-self._max_history_size :]

    def _generate_request_id(
        self,
        requester_id: str,
        changes: list[ConfigChange],
        timestamp: datetime,
    ) -> str:
        data = {
            "requester_id": requester_id,
            "changes": [change.to_dict() for change in changes],
            "timestamp": timestamp.isoformat(),
        }
        hash_input = json.dumps(data, sort_keys=True).encode("utf-8")
        hash_digest = hashlib.sha256(hash_input).hexdigest()
        prefix = timestamp.strftime("%Y%m%d%H%M%S")
        return f"APR-{prefix}-{hash_digest[:8]}"

    def _log_approval_event(self, event_type: str, request: ApprovalRequest) -> None:
        event_data = {
            "event_type": event_type,
            "request_id": request.request_id,
            "requester_id": request.requester_id,
            "requester_name": request.requester_name,
            "approver_id": request.approver_id,
            "approver_name": request.approver_name,
            "status": request.status.value,
            "changes": [change.to_dict() for change in request.changes],
            "rejection_reason": request.rejection_reason,
            "metadata": request.metadata,
        }
        try:
            self._event_store.append_metric(self._bot_id, event_data)
        except Exception:
            logger.debug("Failed to log approval event", exc_info=True)


_global_two_person_rule: TwoPersonRule | None = None


def get_two_person_rule() -> TwoPersonRule:
    global _global_two_person_rule
    if _global_two_person_rule is None:
        _global_two_person_rule = TwoPersonRule()
    return _global_two_person_rule


def create_approval_request(
    requester: User,
    changes: list[ConfigChange],
    *,
    metadata: dict[str, Any] | None = None,
) -> ApprovalRequest:
    return get_two_person_rule().create_approval_request(requester, changes, metadata=metadata)


def approve_request(request_id: str, approver: User) -> tuple[bool, str | None]:
    return get_two_person_rule().approve_request(request_id, approver)


def reject_request(request_id: str, reviewer: User, reason: str) -> tuple[bool, str | None]:
    return get_two_person_rule().reject_request(request_id, reviewer, reason)


def log_config_delta(
    change_type: str,
    changes: dict[str, tuple[Any, Any]],
    *,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    get_two_person_rule().log_config_delta(change_type, changes, user_id=user_id, metadata=metadata)


__all__ = [
    "TwoPersonRule",
    "get_two_person_rule",
    "create_approval_request",
    "approve_request",
    "reject_request",
    "log_config_delta",
]

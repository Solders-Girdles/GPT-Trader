"""
Two-Person Rule for Configuration Changes

Implements two-person approval for critical configuration changes including
risk limits and leverage modifications. All changes are logged to event store.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from bot_v2.persistence.event_store import EventStore
from bot_v2.security.auth_handler import User
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="two_person_rule")


class ChangeType(Enum):
    """Types of configuration changes requiring approval"""

    RISK_LIMIT = "risk_limit"
    LEVERAGE = "leverage"
    POSITION_SIZE = "position_size"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    LIQUIDATION_BUFFER = "liquidation_buffer"
    CIRCUIT_BREAKER = "circuit_breaker"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    SYMBOL_UNIVERSE = "symbol_universe"
    PROFILE_CHANGE = "profile_change"


class ApprovalStatus(Enum):
    """Status of approval request"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    APPLIED = "applied"


@dataclass
class ConfigChange:
    """Configuration change with before/after values"""

    change_type: ChangeType
    field_name: str
    old_value: Any
    new_value: Any
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "change_type": self.change_type.value,
            "field_name": self.field_name,
            "old_value": str(self.old_value),
            "new_value": str(self.new_value),
            "description": self.description,
        }


@dataclass
class ApprovalRequest:
    """Two-person approval request"""

    request_id: str
    requester_id: str
    requester_name: str
    changes: list[ConfigChange]
    status: ApprovalStatus
    created_at: datetime
    expires_at: datetime
    approver_id: str | None = None
    approver_name: str | None = None
    approved_at: datetime | None = None
    rejection_reason: str | None = None
    applied_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if request has expired"""
        return datetime.now(UTC) >= self.expires_at and self.status == ApprovalStatus.PENDING

    @property
    def requires_approval(self) -> bool:
        """Check if request still requires approval"""
        return self.status == ApprovalStatus.PENDING and not self.is_expired

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "request_id": self.request_id,
            "requester_id": self.requester_id,
            "requester_name": self.requester_name,
            "changes": [c.to_dict() for c in self.changes],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "approver_id": self.approver_id,
            "approver_name": self.approver_name,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rejection_reason": self.rejection_reason,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "metadata": self.metadata,
        }


class TwoPersonRule:
    """
    Enforces two-person rule for critical configuration changes.

    Features:
    - Separate requester and approver (cannot be same person)
    - Time-limited approval windows (default 24 hours)
    - Comprehensive audit logging to event store
    - Support for batch changes in single request
    - Automatic expiration of pending requests
    """

    # Critical fields requiring two-person approval
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

    def create_approval_request(
        self,
        requester: User,
        changes: list[ConfigChange],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        """
        Create a new approval request for configuration changes.

        Args:
            requester: User requesting the changes
            changes: List of configuration changes
            metadata: Optional metadata for the request

        Returns:
            ApprovalRequest instance
        """
        with self._lock:
            now = datetime.now(UTC)
            expires_at = now + timedelta(hours=self._approval_timeout_hours)

            # Generate unique request ID
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

            # Log to event store
            self._log_approval_event("approval_request_created", request)

            logger.info(
                f"Created approval request {request_id} by {requester.username}",
                operation="create_approval_request",
                request_id=request_id,
                requester=requester.username,
                change_count=len(changes),
            )

            return request

    def approve_request(
        self,
        request_id: str,
        approver: User,
    ) -> tuple[bool, str | None]:
        """
        Approve a pending request.

        Args:
            request_id: ID of the request to approve
            approver: User approving the request

        Returns:
            Tuple of (success, error_message)
        """
        with self._lock:
            request = self._pending_requests.get(request_id)

            if not request:
                return False, f"Request {request_id} not found"

            # Validate request status
            if request.status != ApprovalStatus.PENDING:
                return False, f"Request is in {request.status.value} status, cannot approve"

            # Check expiration
            if request.is_expired:
                request.status = ApprovalStatus.EXPIRED
                self._log_approval_event("approval_request_expired", request)
                return False, "Request has expired"

            # Validate approver is different from requester
            if approver.id == request.requester_id:
                logger.warning(
                    f"Approval attempt rejected - same user: {approver.username}",
                    operation="approve_request",
                    request_id=request_id,
                    user=approver.username,
                )
                return False, "Approver cannot be the same as requester (two-person rule)"

            # Approve request
            now = datetime.now(UTC)
            request.status = ApprovalStatus.APPROVED
            request.approver_id = approver.id
            request.approver_name = approver.username
            request.approved_at = now

            # Log to event store
            self._log_approval_event("approval_request_approved", request)

            logger.info(
                f"Approved request {request_id} by {approver.username}",
                operation="approve_request",
                request_id=request_id,
                requester=request.requester_name,
                approver=approver.username,
                change_count=len(request.changes),
            )

            return True, None

    def reject_request(
        self,
        request_id: str,
        reviewer: User,
        reason: str,
    ) -> tuple[bool, str | None]:
        """
        Reject a pending request.

        Args:
            request_id: ID of the request to reject
            reviewer: User rejecting the request
            reason: Reason for rejection

        Returns:
            Tuple of (success, error_message)
        """
        with self._lock:
            request = self._pending_requests.get(request_id)

            if not request:
                return False, f"Request {request_id} not found"

            if request.status != ApprovalStatus.PENDING:
                return False, f"Request is in {request.status.value} status, cannot reject"

            # Reject request
            request.status = ApprovalStatus.REJECTED
            request.approver_id = reviewer.id
            request.approver_name = reviewer.username
            request.rejection_reason = reason

            # Move to history
            self._pending_requests.pop(request_id, None)
            self._add_to_history(request)

            # Log to event store
            self._log_approval_event("approval_request_rejected", request)

            logger.info(
                f"Rejected request {request_id} by {reviewer.username}: {reason}",
                operation="reject_request",
                request_id=request_id,
                reviewer=reviewer.username,
                reason=reason,
            )

            return True, None

    def mark_applied(self, request_id: str) -> tuple[bool, str | None]:
        """
        Mark an approved request as applied.

        Args:
            request_id: ID of the request to mark as applied

        Returns:
            Tuple of (success, error_message)
        """
        with self._lock:
            request = self._pending_requests.get(request_id)

            if not request:
                return False, f"Request {request_id} not found"

            if request.status != ApprovalStatus.APPROVED:
                return False, f"Request must be approved before marking as applied"

            # Mark as applied
            request.status = ApprovalStatus.APPLIED
            request.applied_at = datetime.now(UTC)

            # Move to history
            self._pending_requests.pop(request_id, None)
            self._add_to_history(request)

            # Log to event store
            self._log_approval_event("approval_request_applied", request)

            logger.info(
                f"Marked request {request_id} as applied",
                operation="mark_applied",
                request_id=request_id,
            )

            return True, None

    def get_pending_requests(self) -> list[ApprovalRequest]:
        """Get all pending approval requests"""
        with self._lock:
            # Clean up expired requests
            self._cleanup_expired_requests()
            return [r for r in self._pending_requests.values() if r.status == ApprovalStatus.PENDING]

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Get a specific request by ID"""
        with self._lock:
            return self._pending_requests.get(request_id)

    def get_request_history(self, limit: int = 100) -> list[ApprovalRequest]:
        """Get recent request history"""
        with self._lock:
            return self._request_history[-limit:]

    def requires_approval(self, config_changes: dict[str, Any]) -> list[str]:
        """
        Check which config changes require two-person approval.

        Args:
            config_changes: Dictionary of configuration changes (field -> new_value)

        Returns:
            List of field names that require approval
        """
        return [field for field in config_changes.keys() if field in self.CRITICAL_FIELDS]

    def log_config_delta(
        self,
        change_type: str,
        changes: dict[str, tuple[Any, Any]],
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log configuration delta to event store.

        Args:
            change_type: Type of change (e.g., 'config_update', 'risk_limit_change')
            changes: Dictionary of field -> (old_value, new_value) tuples
            user_id: Optional user ID who made the change
            metadata: Optional metadata
        """
        event_data = {
            "event_type": "config_delta",
            "change_type": change_type,
            "changes": {
                field: {"old": str(old), "new": str(new)} for field, (old, new) in changes.items()
            },
            "user_id": user_id,
            "metadata": metadata or {},
        }

        self._event_store.append_metric(self._bot_id, event_data)

        logger.info(
            f"Logged config delta: {change_type}",
            operation="log_config_delta",
            change_type=change_type,
            field_count=len(changes),
        )

    def _cleanup_expired_requests(self) -> None:
        """Clean up expired requests"""
        expired_ids = []

        for request_id, request in self._pending_requests.items():
            if request.is_expired:
                request.status = ApprovalStatus.EXPIRED
                expired_ids.append(request_id)
                self._log_approval_event("approval_request_expired", request)

        for request_id in expired_ids:
            request = self._pending_requests.pop(request_id)
            self._add_to_history(request)

            logger.info(
                f"Expired approval request {request_id}",
                operation="cleanup_expired",
                request_id=request_id,
            )

    def _add_to_history(self, request: ApprovalRequest) -> None:
        """Add request to history"""
        self._request_history.append(request)

        # Trim history if too large
        if len(self._request_history) > self._max_history_size:
            self._request_history = self._request_history[-self._max_history_size :]

    def _generate_request_id(
        self,
        requester_id: str,
        changes: list[ConfigChange],
        timestamp: datetime,
    ) -> str:
        """Generate unique request ID"""
        # Create hash from requester, changes, and timestamp
        data = {
            "requester_id": requester_id,
            "changes": [c.to_dict() for c in changes],
            "timestamp": timestamp.isoformat(),
        }

        hash_input = json.dumps(data, sort_keys=True).encode()
        hash_digest = hashlib.sha256(hash_input).hexdigest()

        # Use timestamp prefix for readability
        timestamp_prefix = timestamp.strftime("%Y%m%d%H%M%S")
        return f"APR-{timestamp_prefix}-{hash_digest[:8]}"

    def _log_approval_event(self, event_type: str, request: ApprovalRequest) -> None:
        """Log approval event to event store"""
        event_data = {
            "event_type": event_type,
            "request_id": request.request_id,
            "requester_id": request.requester_id,
            "requester_name": request.requester_name,
            "approver_id": request.approver_id,
            "approver_name": request.approver_name,
            "status": request.status.value,
            "changes": [c.to_dict() for c in request.changes],
            "rejection_reason": request.rejection_reason,
            "metadata": request.metadata,
        }

        self._event_store.append_metric(self._bot_id, event_data)


# Global instance
_two_person_rule: TwoPersonRule | None = None


def get_two_person_rule() -> TwoPersonRule:
    """Get the global two-person rule instance"""
    global _two_person_rule
    if _two_person_rule is None:
        _two_person_rule = TwoPersonRule()
    return _two_person_rule


# Convenience functions
def create_approval_request(
    requester: User,
    changes: list[ConfigChange],
    *,
    metadata: dict[str, Any] | None = None,
) -> ApprovalRequest:
    """Create approval request"""
    return get_two_person_rule().create_approval_request(requester, changes, metadata=metadata)


def approve_request(request_id: str, approver: User) -> tuple[bool, str | None]:
    """Approve request"""
    return get_two_person_rule().approve_request(request_id, approver)


def log_config_delta(
    change_type: str,
    changes: dict[str, tuple[Any, Any]],
    *,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log configuration delta"""
    get_two_person_rule().log_config_delta(change_type, changes, user_id=user_id, metadata=metadata)
